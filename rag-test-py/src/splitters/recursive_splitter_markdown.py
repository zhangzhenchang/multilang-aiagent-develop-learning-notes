"""recursive_splitter_markdown.py - Markdown 文本分割示例"""
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter


README_TEXT = """# Project Name

> A brief description of your project

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- ✨ Feature 1
- 🚀 Feature 2
- 💡 Feature 3

## Installation

```bash
npm install project-name
```

## Usage

### Basic Usage

```javascript
import { Project } from 'project-name';

const project = new Project();
project.init();
```

### Advanced Usage

```javascript
const project = new Project({
  config: {
    apiKey: 'your-api-key',
    timeout: 5000,
  }
});

await project.run();
```

## API Reference

### `Project`

Main class for the project.

#### Methods

- `init()`: Initialize the project
- `run()`: Run the project
- `stop()`: Stop the project

## Contributing

Contributions are welcome! Please read our [contributing guide](CONTRIBUTING.md).

## License

MIT License"""


def main() -> None:
    document = Document(page_content=README_TEXT)
    splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=80)
    split_documents = splitter.split_documents([document])

    for item in split_documents:
        print(item)
        print("charater length:", len(item.page_content))


if __name__ == "__main__":
    main()
