from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    input = sys.stdin.readline
    T = int(input().strip())
    for _ in range(T):
        n, m = map(int, input().split())
        wanted = list(map(int, input().split()))

        size = n + m + 5  
        seg = SegmentTree(
            size=size,
            default_value=0,
            merge_func=lambda a, b: a+b,
            convert_func=lambda x: x
        )

        for i in range(n):
            seg.update_pos(i, 1)  # i 위치에 DVD 존재
        pos = [0]*(n+1)  # pos[i] = 현재 DVD i의 위치
        for i in range(1, n+1):
            pos[i] = i-1  # 초기 위치

        top_idx = -1

        for dvd in wanted:
            curr_pos = pos[dvd]
            above_count = 0
            if curr_pos > 0:
                above_count = seg.range_query(0, curr_pos-1)

            print(above_count, end=' ')

            seg.update_pos(curr_pos, 0)  # 기존 위치에서 제거
            seg.update_pos(top_idx, 1)   # top_idx 위치에 DVD 놓음
            pos[dvd] = top_idx
            top_idx -= 1

        print() 


if __name__ == "__main__":
    main()