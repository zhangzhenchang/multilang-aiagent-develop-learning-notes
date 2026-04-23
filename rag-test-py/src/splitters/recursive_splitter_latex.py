"""recursive_splitter_latex.py - LaTeX 文本分割示例"""
from langchain_core.documents import Document
from langchain_text_splitters import LatexTextSplitter


LATEX_TEXT = r"""\int x^{\mu}\mathrm{d}x=\frac{x^{\mu +1}}{\mu +1}+C, \left({\mu \neq -1}\right) \int \frac{1}{\sqrt{1-x^{2}}}\mathrm{d}x= \arcsin x +C \int \frac{1}{\sqrt{1-x^{2}}}\mathrm{d}x= \arcsin x +C \begin{pmatrix}  
  a_{11} & a_{12} & a_{13} \\
  a_{21} & a_{22} & a_{23} \\
  a_{31} & a_{32} & a_{33}  
\end{pmatrix} """


def main() -> None:
    document = Document(page_content=LATEX_TEXT)
    splitter = LatexTextSplitter(chunk_size=200, chunk_overlap=40)
    split_documents = splitter.split_documents([document])

    for item in split_documents:
        print(item)
        print("charater length:", len(item.page_content))


if __name__ == "__main__":
    main()
