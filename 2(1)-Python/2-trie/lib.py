from dataclasses import dataclass, field
from typing import Generic, Optional, Iterable, List, TypeVar
from math import factorial

T = TypeVar("T", bound=str)  # 여기서는 문자열 전용 Trie로 사용한다고 가정


@dataclass
class TrieNode:
    """
    body: 현재 노드가 나타내는 문자(코드포인트) 
    children: 자식 노드들의 인덱스 리스트
    is_end: 어떤 단어의 끝인지 여부
    """
    body: Optional[int] = None
    children: List[int] = field(default_factory=list)
    is_end: bool = False


class Trie(Generic[T], List[TrieNode]):
    def __init__(self) -> None:
        super().__init__()
        # 루트 노드를 하나 미리 생성 (body=None)
        self.append(TrieNode(body=None))

    def push(self, seq: T) -> None:
        """
        seq: 문자열 (str)
        Trie에 seq를 삽입
        """
        current_index = 0  # 루트 노드부터 시작
        for ch in seq:
            found = False
            for child_index in self[current_index].children:
                # 저장은 ord(ch)로 했으므로, 비교할 때도 ord(ch)로 비교
                if self[child_index].body == ord(ch):
                    current_index = child_index
                    found = True
                    break

            # 기존 자식 중에 없으면 새로 만든다
            if not found:
                new_node = TrieNode(body=ord(ch))
                self.append(new_node)
                new_index = len(self) - 1
                self[current_index].children.append(new_index)
                current_index = new_index

        self[current_index].is_end = True

    def count_buttons(self, word: T) -> int:
        """
        word: 문자열
        사용자가 word를 입력할 때 버튼을 눌러야 하는 횟수를 리턴
        """
        node_index = 0  # 루트 노드의 인덱스
        cnt = 0

        for i, ch in enumerate(word):
            # 1) 루트 노드에서 첫 글자를 입력할 때는 무조건 버튼 클릭
            if node_index == 0:
                cnt += 1
            # 2) 자식이 여러 개 존재하거나, 이 노드 자체가 단어의 끝이라면
            #    자동입력이 불가능하므로 버튼을 눌러야 함
            elif len(self[node_index].children) != 1 or self[node_index].is_end:
                cnt += 1

            # 다음 글자에 해당하는 자식 노드를 찾는다
            next_index = None
            for child_idx in self[node_index].children:
                if self[child_idx].body == ord(ch):
                    next_index = child_idx
                    break

            if next_index is None:
                # Trie 안에 없는 글자를 입력하려고 한 경우
                raise ValueError(f"Character '{ch}' not found in Trie.")

            node_index = next_index

        return cnt
