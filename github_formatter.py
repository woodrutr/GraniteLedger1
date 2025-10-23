from pathlib import Path
import re


def parser(markdown):
    """Parses original markdown formatting into Github-compatible version

    Parameters
    ----------
    markdown : Path
        Path object pointing to markdown file
    """
    with open(Path('.', markdown), encoding='utf8') as f:
        math = False
        i = 0

        lines = f.readlines()
        newlines = []
        for line in lines:
            if re.search(pattern=r'\$\$', string=line) and not re.search(
                pattern=r'\$\$(.*)\$\$', string=line
            ):
                if not math:
                    math = True

                    # strip all non-dollar text
                    newline = line.replace('$$', '').replace('\n', '').strip()
                    newline = ['\n', '$$\n', '\\begin{aligned}\n', newline]
                    for each in newline:
                        newlines.append(each)
                else:
                    math = False

                    #
                    newline = ['\\end{aligned}\n', '$$\n', '\n']
                    for each in newline:
                        newlines.append(each)
            else:
                if math:
                    newline = line.replace('sw_', 'sw \\_ ')
                    newline = re.sub(r'\\tag\{(.+)\}', '', newline)
                    newline = re.sub(r'_(?<!\\)', r'\\_', newline)
                    newline = re.sub(r'\|', r'\\|', newline)
                    newline = re.sub(r'<', r' \\< ', newline)
                    newline = re.sub(r'>', r' \\> ', newline)
                    if not newline.strip() == '':
                        newlines.append(newline)
                else:
                    newlines.append(line)
            i += 1
    f.close()

    ### write to new file
    with open(Path('.', markdown), 'w', encoding='utf8') as f:
        f.writelines(newlines)
        f.close()
    return


if __name__ == '__main__':
    ### List all markdown files
    wd = Path('.')

    files = [x for x in wd.glob(pattern='**/README.md')]
    [x for x in map(parser, files)]
