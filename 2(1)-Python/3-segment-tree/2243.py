from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    input = sys.stdin.readline
    n = int(input().strip())  
    MAX_TASTE = 1_000_000

    seg = SegmentTree(
        size=MAX_TASTE,
        default_value=0,
        merge_func=lambda a, b: a + b,
        convert_func=lambda x: x  
    )

    for _ in range(n):
        line = input().split()
        A = int(line[0])
        if A == 1:
            B = int(line[1])
            taste_idx = seg.kth_value(B)  
            real_taste = taste_idx + 1

            # 사탕 한 개 제거 (count -= 1)
            prev_count = seg.range_query(taste_idx, taste_idx)  # 현재 맛 개수
            seg.update_pos(taste_idx, prev_count - 1)

            print(real_taste)

        else:  # A == 2
            B = int(line[1])  # 맛
            C = int(line[2])  # 개수 증감
            idx = B-1
            curr = seg.range_query(idx, idx)
            seg.update_pos(idx, curr + C)

    pass


if __name__ == "__main__":
    main()