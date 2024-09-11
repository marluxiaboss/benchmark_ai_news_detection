# Welcome to My Project

## Introduction

Welcome to the documentation for My Project. This is where you can find all the information you need.

## Features

### Feature 1

Details about feature 1.

### Feature 2

Details about feature 2.

## Documentation

```{toctree}
---
maxdepth: 2
caption: Getting started
---
subpages/getting_started.md
```

```{toctree}
---
maxdepth: 3
caption: Contents
---
subpages/how_to_generate.md
subpages/how_to_add_index.md
subpages/details.md
```

## How-To Guides

```{note}
This is a note created with MyST.
```

### 4. **Card Directive Options**

- `:link:` - The URL that the card should link to. When a user clicks on the card, they will be directed to this URL.
- `:title:` - The title of the card. This appears prominently on the card.
- `:icon:` - (Optional) An icon or emoji to display on the card.
- The content below the directive is the body of the card, where you can include additional text, images, or Markdown elements.

### 5. **Example Markdown File**

Here's a full example of a Markdown file with cards:

```{card} Card Title
**Header**: This is the header

This is the card content

**Footer**: This is the footer
```

Explore more about this topic and find detailed information on the website.