"""已分配地址整批迁移接口的端到端测试。

直接运行：``python3 -m tests.test_relocate``（或 pytest 收集）。

造数：父网段 10.10.0.0/16，初始保留 10.10.0.0/20 与 10.10.255.0/24；
先在池尾依次提交两个 /25（铺满 10.10.254.0/24），再在 10.10.252.0/23
起点依次提交三个 /28：

    10.10.254.0/25     = 254.0..127
    10.10.254.128/25   = 254.128..255
    10.10.252.0/28     = 252.0..15
    10.10.252.16/28    = 252.16..31
    10.10.252.32/28    = 252.32..47

覆盖：
* 非法输入 422：非规范写法 / 无法解析、两端地址族不同、前缀长度不同、
  来源之间重叠、目标之间重叠、空 moves 列表；
* 冲突 409：版本过期、来源未被已分配地址完整覆盖（含已分配集为空）、
  目标越出父网段、目标触及保留区、目标触及未迁移的已分配地址；
* 409 不改动池（版本、已分配集、空闲 CIDR 全部保持不变）；
* 整批同时迁移：两组 /25 相互交换（目标使用另一组来源释放的地址）；
* 单组迁移到空闲区后，最少规范 CIDR 重写且池状态立即反映；
* 多组迁移后相邻目标合并重写（两个 /28 -> /27）；
* 目标与剩余已分配地址相邻时合并重写（/27 + /28 -> /26）；
* 来源与目标相同的空操作仍自增版本。
"""
import ipaddress
import os
import tempfile
import threading

os.environ["PARENT_CIDR"] = "10.10.0.0/16"
os.environ["RESERVED_CIDRS"] = "10.10.0.0/20,10.10.255.0/24"
_tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_tmp, "test_relocate.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

SEED = [
    "10.10.254.0/25",
    "10.10.254.128/25",
    "10.10.252.0/28",
    "10.10.252.16/28",
    "10.10.252.32/28",
]


def _pool():
    return client.get("/api/pool").json()


def _preview(prefixes):
    r = client.post(
        "/api/allocations/preview", json={"prefix_lengths": prefixes}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True, body
    return body


def _commit(version, allocations):
    r = client.post(
        "/api/allocations/commit",
        json={"pool_version": version, "allocations": allocations},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _relocate(version, moves):
    return client.post(
        "/api/allocations/relocate",
        json={"pool_version": version, "moves": moves},
    )


def _move(src, dst):
    return {"from_cidr": src, "to_cidr": dst}


def _numsorted(cidrs):
    return sorted(
        cidrs, key=lambda c: int(ipaddress.ip_network(c, strict=False).network_address)
    )


def main():
    assert _pool()["version"] == 1

    # ---- 已分配集为空：迁移 -> 409，池不变 ----
    r = _relocate(1, [_move("10.10.254.0/25", "10.10.200.0/25")])
    assert r.status_code == 409, r.text
    assert "not fully covered" in r.json()["detail"]
    assert _pool()["version"] == 1
    assert _pool()["allocated_cidrs"] == []
    print("empty pool relocate -> 409 ok")

    # ---- 造数：/25, /25, /28, /28, /28 ----
    version = 1
    for plen in (25, 25, 28, 28, 28):
        got = _preview([plen])["allocations"][0]
        _commit(version, [got])
        version += 1
    assert version == 6
    assert _numsorted(_pool()["allocated_cidrs"]) == _numsorted(SEED)
    print("seed allocations:", SEED)

    # ---- 非法输入 -> 422，池状态与版本不变 ----
    bad_bodies = [
        # 主机位非零
        {"pool_version": 6, "moves": [_move("10.10.252.1/28", "10.10.253.0/28")]},
        {"pool_version": 6, "moves": [_move("10.10.252.0/28", "10.10.253.1/28")]},
        # 无法解析
        {"pool_version": 6, "moves": [_move("not-a-cidr", "10.10.253.0/28")]},
        # 两端地址族不同
        {"pool_version": 6, "moves": [_move("10.10.252.0/28", "2001:db8::/28")]},
        # 两端前缀长度不同
        {"pool_version": 6, "moves": [_move("10.10.252.0/28", "10.10.252.64/27")]},
        # 来源之间重叠（10.10.252.0/29 落在另一来源 10.10.252.0/28 内）
        {"pool_version": 6, "moves": [
            _move("10.10.252.0/28", "10.10.200.0/28"),
            _move("10.10.252.0/29", "10.10.201.0/29"),
        ]},
        # 目标之间重叠
        {"pool_version": 6, "moves": [
            _move("10.10.252.0/28", "10.10.200.0/28"),
            _move("10.10.252.16/28", "10.10.200.8/29"),
        ]},
        # 空 moves 列表
        {"pool_version": 6, "moves": []},
    ]
    for body in bad_bodies:
        r = client.post("/api/allocations/relocate", json=body)
        assert r.status_code == 422, (body, r.text)
    assert _pool()["version"] == 6
    assert _numsorted(_pool()["allocated_cidrs"]) == _numsorted(SEED)
    print("invalid bodies -> 422 ok")

    # ---- 冲突场景：池在每一步后均保持不变 ----
    conflict_cases = [
        (1, [_move("10.10.254.0/25", "10.10.200.0/25")], "version changed"),
        # 来源完全空闲
        (6, [_move("10.10.252.48/28", "10.10.253.0/28")], "not fully covered"),
        # 来源一半已分配一半空闲
        (6, [_move("10.10.252.32/27", "10.10.253.0/27")], "not fully covered"),
        # 目标越出父网段
        (6, [_move("10.10.252.0/28", "10.11.0.0/28")], "outside of pool"),
        # 目标触及保留区
        (6, [_move("10.10.252.0/28", "10.10.0.0/28")], "reserved"),
        # 目标触及未被本批释放的已分配地址
        (6, [_move("10.10.252.0/28", "10.10.252.16/28")], "not released"),
    ]
    for ver, moves, needle in conflict_cases:
        before = _pool()
        r = _relocate(ver, moves)
        assert r.status_code == 409, (moves, r.text)
        assert needle in r.json()["detail"], (needle, r.json()["detail"])
        after = _pool()
        assert after["version"] == before["version"]
        assert after["allocated_cidrs"] == before["allocated_cidrs"]
        assert after["free_cidrs"] == before["free_cidrs"]
    print("conflicts -> 409, pool untouched")

    # ---- 整批同时迁移：两个 /25 相互交换（目标落在对方释放区）----
    # 两个相邻 /25 目标覆盖整个 10.10.254.0/24，重写时合并为单个 /24；
    # 三个 /28（252.0..47，共 48 个地址）的最少规范 CIDR 为 /27 + /28。
    r = _relocate(6, [
        _move("10.10.254.0/25", "10.10.254.128/25"),
        _move("10.10.254.128/25", "10.10.254.0/25"),
    ])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "relocated"
    assert body["pool_version"] == 7
    assert _numsorted(body["allocated_cidrs"]) == _numsorted([
        "10.10.254.0/24",
        "10.10.252.0/27",
        "10.10.252.32/28",
    ]), body["allocated_cidrs"]
    pool7 = _pool()
    assert body["free_cidrs"] == pool7["free_cidrs"]
    print("simultaneous swap ok")

    # ---- 单组迁移到空闲区：10.10.254.0/25 -> 10.10.200.0/25 ----
    # 来源是 /24 内的一个 /25，扣除后 10.10.254.0/24 余下的 /25 保持
    # 单条规范 CIDR；252.0..47 的 /27 + /28 划分不变。
    r = _relocate(7, [_move("10.10.254.0/25", "10.10.200.0/25")])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_version"] == 8
    assert _numsorted(body["allocated_cidrs"]) == _numsorted([
        "10.10.200.0/25",
        "10.10.252.0/27",
        "10.10.252.32/28",
        "10.10.254.128/25",
    ]), body["allocated_cidrs"]
    pool8 = _pool()
    assert body["free_cidrs"] == pool8["free_cidrs"]
    # 迁出地址立即空闲，落入地址立即占用。
    assert "10.10.254.0/25" in pool8["free_cidrs"], pool8["free_cidrs"]
    assert "10.10.200.0/25" not in pool8["free_cidrs"]
    print("single move into free space ok")

    # ---- 多组迁移：两个 /28 搬到 10.10.253 起点，相邻目标合并为 /27 ----
    # 来源 252.0/28、252.16/28 恰好覆盖剩余的 252.0/27，整块消失。
    r = _relocate(8, [
        _move("10.10.252.0/28", "10.10.253.0/28"),
        _move("10.10.252.16/28", "10.10.253.16/28"),
    ])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_version"] == 9
    assert _numsorted(body["allocated_cidrs"]) == _numsorted([
        "10.10.200.0/25",
        "10.10.252.32/28",
        "10.10.253.0/27",
        "10.10.254.128/25",
    ]), body["allocated_cidrs"]
    print("adjacent targets collapse into minimal CIDRs ok")

    # ---- 目标与剩余已分配地址相邻：新 /28 紧邻 253.0/27 ----
    # 48 个连续地址（253.0..47）的最少规范 CIDR 为 /27 + /28。
    r = _relocate(9, [_move("10.10.252.32/28", "10.10.253.32/28")])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_version"] == 10
    assert _numsorted(body["allocated_cidrs"]) == _numsorted([
        "10.10.200.0/25",
        "10.10.253.0/27",
        "10.10.253.32/28",
        "10.10.254.128/25",
    ]), body["allocated_cidrs"]
    print("target merges with remaining allocation ok")

    # ---- 空操作（来源即目标）：合法，版本仍自增 ----
    r = _relocate(10, [_move("10.10.200.0/25", "10.10.200.0/25")])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_version"] == 11
    assert _numsorted(body["allocated_cidrs"]) == _numsorted([
        "10.10.200.0/25",
        "10.10.253.0/27",
        "10.10.253.32/28",
        "10.10.254.128/25",
    ])
    print("no-op move ok")

    # ---- 同版本并发迁移：写锁互斥，恰有一个成功 ----
    # 当前已分配 10.10.253.0/27 + 10.10.253.32/28（覆盖 10.10.253.0/26）。
    # 两线程用同一版本把同一来源迁往不同空闲目标，恰有一个 200。
    v11 = _pool()["version"]
    assert v11 == 11
    statuses: list[int] = []
    barrier = threading.Barrier(2)

    def _worker(dst: str) -> None:
        barrier.wait()
        statuses.append(
            _relocate(11, [_move("10.10.253.0/27", dst)]).status_code
        )

    t1 = threading.Thread(target=_worker, args=("10.10.200.128/27",))
    t2 = threading.Thread(target=_worker, args=("10.10.201.0/27",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert sorted(statuses) == [200, 409], statuses
    final = _pool()
    assert final["version"] == 12
    # 未迁移的 10.10.253.32/28 保留；恰有一个目标落库。
    landed = set(final["allocated_cidrs"]) & {
        "10.10.200.128/27",
        "10.10.201.0/27",
    }
    assert len(landed) == 1, final["allocated_cidrs"]
    print("concurrent relocate: exactly one succeeds ok")

    print("ALL RELOCATE TESTS PASSED")


if __name__ == "__main__":
    main()
