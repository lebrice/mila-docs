# Contributing to the Mila Docs

Thank you for your interest into making a better documentation for all at Mila.

Here are some guidelines to help bring your contributions to life.

## What should be included in the Mila Docs

* Mila cluster usage
* Digital Research Alliance of Canada cluster usage
* Job management tips / tricks
* Research good practices
* Software development good practices
* Useful tools

**_NOTE_**: Examples should aim to not consume much more than 1 GPU/hour and 2 CPU/hour

## Issues / Pull Requests

### Issues

Issues can be used to report any error in the documentation, missing or unclear
sections, broken tools or other suggestions to improve the overall
documentation.

### Pull Requests

PRs are welcome and we value the contents of contributions over the appearance
or functionality of the pull request. If you don't know how to write the proper
markup in reStructuredText, simply provide the content you would like to add in
the PR text form which supports markdown or with instructions to format the
content. In the PR, reference the related issues like this:

```
Resolves: #123
See also: #456, #789
```

If you would like to contribute directly in the code of the documentation, keep
the lines width to 80 characters or less. You can attempt to build the docs
yourself to see if the formatting is right:

#### Install dependencies (with uv)
```console
uv pip install -r docs/requirements.txt
uv pip install mkdocs mkdocs-material pre-commit
pre-commit install
```

#### Build and serve the documentation
```console
mkdocs serve
```

This will start a local server (by default at http://127.0.0.1:8000/) where you can preview the documentation live as you edit Markdown files.

If you have any trouble building the docs, don't hesitate to open an issue to
request help.

Regarding the restructured text format, you can simply provide the content
you would like to add in markdown or plain text format if more convenient
for you and someone down the line should take responsibility to convert
the format.

## Sphinx / reStructuredText (reST)

The markup language used for the Mila Docs is
[reStructuredText](http://docutils.sourceforge.net/rst.html) and we follow the
[Python’s Style Guide for documenting](https://docs.python.org/devguide/documenting.html#style-guide).

Here are some of reST syntax directives which are useful to know :
(more can be found in
[Sphinx's reST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)):


### Inline markup

* one asterisk: `*text*` for *emphasis* (italics),
* two asterisks: `**text**` for **strong emphasis** (boldface), and
* backquotes: ` ``text`` ` for `code samples`, and
* external links: `` `Link text <http://target>`_ ``.

### Lists

```reST
* this is
* a list

  * with a nested list
  * and some subitems

* and here the parent list continues
```

### Sections

```reST
#################
This is a heading
#################
```

There are no heading levels assigned to certain characters as the structure is
determined from the succession of headings. However, the Python documentation
suggests the following convention:

    * `#` with overline, for parts
    * `*` with overline, for chapters
    * `=`, for sections
    * `-`, for subsections
    * `^`, for subsubsections
    * `"`, for paragraphs

### Note box

```reST
.. note:: This is a long
   long long note
```

### Collapsible boxes

This is a local extension, not part of Sphinx itself.  It works like this:

```reST
.. container:: toggle

    .. container:: header

        **Show/Hide Code**

    .. code-block:: <type>
       ...
```