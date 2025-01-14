import sys
from lib import Trie


def main() -> None:
    data = sys.stdin.read().splitlines()
    index = 0
    results = []

    while index < len(data):
        # 공백 줄은 건너뛰기
        while index < len(data) and not data[index].strip():
            index += 1
        if index >= len(data):
            break

        # N(단어 개수) 읽기
        try:
            N = int(data[index].strip())
        except ValueError:
            index += 1
            continue
        index += 1

        words = []
        for _ in range(N):
            if index >= len(data):
                break
            word = data[index].strip()
            if word:
                words.append(word)
            index += 1

        # Trie 생성
        word_trie: Trie[str] = Trie()
        for w in words:
            word_trie.push(w)

        # 각 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
        total_clicks = 0
        for w in words:
            total_clicks += word_trie.count_buttons(w)

        # 평균 계산
        average_clicks = total_clicks / len(words)
        results.append(f"{average_clicks:.2f}")

    print("\n".join(results))


if __name__ == "__main__":
    main()
