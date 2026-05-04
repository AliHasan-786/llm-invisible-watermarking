import asyncio, sys
from playwright.async_api import async_playwright

async def main(html_path, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f'file://{html_path}')
        await page.emulate_media(media='print')
        await page.add_style_tag(content='''
            @page { size: Letter; margin: 0.40in; }
            body { font-size: 10pt; line-height: 1.22; }
            h1 { font-size: 15pt; margin: 0.2em 0 0.2em; }
            h2 { font-size: 11.5pt; margin: 0.4em 0 0.2em; }
            h3 { font-size: 10.5pt; margin: 0.3em 0 0.15em; }
            div.jp-Cell { margin: 0.05em 0; padding: 0; }
            div.jp-MarkdownOutput, div.jp-RenderedHTMLCommon { margin: 0.1em 0; }
            div.jp-Notebook { padding: 0; }
            div.jp-CodeMirrorEditor { padding: 0.15em 0.25em; }
            pre, code, .CodeMirror { font-size: 8pt !important; line-height: 1.10 !important; }
            div.jp-OutputArea-output, div.jp-RenderedText { padding: 0; margin: 0.05em 0; font-size: 8pt; line-height: 1.10; }
            img { max-width: 88%; }
            div.jp-OutputArea-prompt, div.jp-InputPrompt { display: none !important; }
            div.jp-CodeCell { margin: 0.05em 0; padding: 0; }
            ul, ol { margin: 0.2em 0; padding-left: 1.2em; }
            li { margin: 0.05em 0; line-height: 1.22; }
            p { margin: 0.18em 0; }
            div.jp-RenderedMarkdown pre { margin: 0.15em 0; }
            div.highlight pre { margin: 0; padding: 0.15em 0.3em; }
        ''')
        await page.pdf(path=pdf_path, format='Letter', print_background=True,
                       margin={'top':'0.40in','bottom':'0.40in','left':'0.40in','right':'0.40in'})
        await browser.close()

asyncio.run(main(sys.argv[1], sys.argv[2]))
