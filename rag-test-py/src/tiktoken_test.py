"""tiktoken_test.py - 演示不同文本的 token 长度"""
import tiktoken


def main() -> None:
    model_name = "gpt-4"
    encoding = tiktoken.encoding_for_model(model_name)
    print(encoding.name)

    enc = tiktoken.get_encoding("cl100k_base")
    print("apple", len(enc.encode("apple")))
    print("pineapple", len(enc.encode("pineapple")))
    print("苹果", len(enc.encode("苹果")))
    print("吃饭", len(enc.encode("吃饭")))
    print("一二三", len(enc.encode("一二三")))


if __name__ == "__main__":
    main()
