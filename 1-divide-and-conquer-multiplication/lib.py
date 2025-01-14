from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        matrix의 key 위치 원소를 value를 1000으로 나눈 나머지로 수정. 
        
        Args:
            - key (tuple[int, int]): matrix 내 원소 위치를 지시.
            - value (int): key 위치 원소의 원래 값.
        
        Returns:
            - None: 리턴하지 않음.
                - 행렬 값을 수정할 뿐.
        """
        self.matrix[key[0]][key[1]] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        분할 정복을 이용하여 행렬을 n번 제곱한 결과를 반환.
        
        Args:
            - n (int): 지수
            
        Returns:
            - Matrix: self^n 결과 행렬
        """
        result = Matrix.eye(self.shape[0])
        base = self.clone()
        
        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2
            
        return result

    def __repr__(self) -> str:
        """
        행렬을 문자열로 반환
        행은 \n으로 구분, 원소는 공백으로 구분.
        
        Returns:
            - str: 행렬의 각 원소를 1000으로 나눈 나머지로 출력한 문자열
        """
        
        lines = []
        for row in self.matrix:
            row_str = " ".join(str(x % self.MOD) for x in row)
            lines.append(row_str)
        return "\n".join(lines)