from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, cast


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    
    """ 
    
    SegmentTree class.

    Attributes: 
        - size (int): 데이터의 크기. 
        - default_value (T): 기본값. 
        - merge_func (Callable[[T, T], T]): 두 노드 값을 병합하여 하나의 노드 값으로 만드는 함수. 
        - convert_func (Callable[[U], T]): 외부 입력 값(U)을 세그먼트 트리 노드 자료형(T)으로 변환하는 함수. 
        - tree (list): 내부 세그먼트 트리 구조를 저장하는 리스트.

    Methods: 
        - init_build(arr): arr를 기반으로 세그먼트 트리를 초기화. 
        - update_pos(pos, val): pos 위치의 값을 val로 갱신. 
        - range_query(left, right): [left..right] 구간을 병합한 결과 반환. 
        - kth_value(k): 누적 값이 k가 되는 위치를 찾아 반환. 
        
    """
    def __init__(
        self,
        size: int,
        default_value: T,
        merge_func: Callable[[T, T], T],
        convert_func: Optional[Callable[[U], T]] = None
    ) -> None:
        """
        SegmentTree constructor.

        Args:
            size (int): The size of the array we want to manage (0-based indexing assumed).
            default_value (T): The "identity" or default node value for out-of-range queries.
            merge_func (Callable[[T, T], T]): A function that merges two child node values into one.
            convert_func (Callable[[U], T], optional): Converts an external input value U into a node value T. Defaults to identity if U == T.
        
        Internal:
            self.tree (list): The segment tree structure, sized up to 4*size.
        """
        def local_default_convert(x: U) -> T:
            return cast(T, x)

        if convert_func is None:
            convert_func = local_default_convert
        
        self.size: int = size
        self.default_value: T = default_value
        self.merge_func: Callable[[T, T], T] = merge_func
        self.convert_func: Callable[[U], T] = convert_func
        self.tree = [default_value] * (4 * size)

    def build(self, arr, idx, start, end) -> None:
        """
        재귀적으로 [start..end] 구간을 세그먼트 트리에 구성한다.

        Args:
            arr (list[U]): 원본 데이터.
            idx (int): 현재 세그먼트 트리 노드 인덱스.
            start (int): 구간 시작 인덱스
            end (int): 구간 끝 인덱스

        Post-condition:
            self.tree[idx]에 구간 [start..end]를 대표하는 노드 값을 저장한다.
        """
        if start == end:
            self.tree[idx] = self.convert_func(arr[start])
            return
        mid = (start + end) // 2
        self.build(arr, idx*2, start, mid)
        self.build(arr, idx*2+1, mid+1, end)
        self.tree[idx] = self.merge_func(self.tree[idx*2], self.tree[idx*2+1])

    def init_build(self, arr) -> None:
        """
        외부에서 배열 arr를 이용해 트리를 초기화하는 함수.

        Args:
            arr (list[U]): 길이 self.size 이상인 배열.
        
        """
        if self.size > 0:
            self.build(arr, 1, 0, self.size - 1)

    def update(self, idx, start, end, pos, val) -> None:
        """
        재귀적으로 arr[pos] 위치를 val로 업데이트.

        Args:
            idx (int): 세그먼트 트리 노드 인덱스
            start (int): 현재 노드가 커버하는 구간의 시작
            end (int): 현재 노드가 커버하는 구간의 끝
            pos (int): 업데이트할 배열 인덱스
            val (U): 업데이트할 값 
        """
        if pos < start or pos > end:
            return  # out of range
        if start == end:
            # leaf node
            self.tree[idx] = self.convert_func(val)
            return
        mid = (start + end) // 2
        self.update(idx*2, start, mid, pos, val)
        self.update(idx*2+1, mid+1, end, pos, val)
        self.tree[idx] = self.merge_func(self.tree[idx*2], self.tree[idx*2+1])

    def update_pos(self, pos, val) -> None:
        """
        pos 위치의 데이터를 val로 업데이트.

        Args:
            pos (int): 업데이트할 배열 인덱스 
            val (U): 새로 대입할 값
        """
        if self.size == 0:
            return
        self.update(1, 0, self.size - 1, pos, val)

    def query(self, idx, start, end, left, right):
        """
        재귀적으로 구간 [left..right]의 정보를 구한다.

        Args:
            idx (int): 세그먼트 트리 노드 인덱스
            start (int): 현재 노드의 구간 시작
            end (int): 현재 노드의 구간 끝
            left (int): 쿼리 구간 시작
            right (int): 쿼리 구간 끝

        Returns:
            T: 구간 [left..right]가 커버하는 데이터 병합 결과
        """
        if right < start or end < left:
            return self.default_value
        if left <= start and end <= right:
            return self.tree[idx]
        mid = (start + end) // 2
        lres = self.query(idx*2, start, mid, left, right)
        rres = self.query(idx*2+1, mid+1, end, left, right)
        return self.merge_func(lres, rres)

    def range_query(self, left, right):
        """
        [left..right] 구간에 대한 쿼리리를 수행.

        Args:
            left (int): 쿼리 구간 시작 
            right (int): 쿼리 구간 끝

        Returns:
            T: 병합 결과 
        """
        if self.size == 0 or left > right:
            return self.default_value
        return self.query(1, 0, self.size - 1, left, right)

    def find_kth(self, idx, start, end, k):
        """
        합 기반 세그먼트 트리에서 'k번째 원소'의 인덱스를 찾는 내부 함수.

        Args:
            idx (int): 세그먼트 트리 노드 인덱스
            start (int): 현재 노드 구간 시작
            end (int): 현재 노드 구간 끝
            k (int): 'k번째' 위치를 찾고자 하는 기준

        Returns:
            int: k번째가 위치한 인덱스 (start..end 내)
        """
        if start == end:
            return start
        left_val = self.tree[idx*2]
        mid = (start + end) // 2

        if left_val >= k:
            return self.find_kth(idx*2, start, mid, k)
        else:
            return self.find_kth(idx*2+1, mid+1, end, k - left_val)

    def kth_value(self, k):
        """
        합 기반 세그먼트 트리에서 k번째 원소가 위치한 인덱스를 찾는다.

        Args:
            k (int): 1-based 혹은 0-based k

        Returns:
            int: 세그먼트 트리에서 k번째를 만족하는 인덱스
        
        """
        if self.size == 0 or k <= 0:
            return 0 
        return self.find_kth(1, 0, self.size - 1, k)


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