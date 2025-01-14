from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    input = sys.stdin.readline
    N = int(input())
    arr = list(map(int, input().split()))
    M = int(input().strip())

    # 세그먼트 트리 생성 (Pair 기반)
    seg = SegmentTree[Pair,int](
        size=N,
        default_value=Pair.default(),
        merge_func=Pair.f_merge,
        convert_func=Pair.f_conv
    )

    # build
    seg.init_build(arr)

    for _ in range(M):
        line = input().split()
        op = int(line[0])
        if op == 1:
            # 1 i v
            i = int(line[1]) - 1  # 1-based -> 0-based
            v = int(line[2])
            # update
            seg.update_pos(i, v)
        else:
            # 2 l r
            l = int(line[1]) - 1
            r = int(line[2]) - 1
            node_val = seg.range_query(l, r)
            print(node_val.sum())



if __name__ == "__main__":
    main()