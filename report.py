
"""
Generate a report of the Bluetooth tracking data.
Input data from wearables and GT data if anliable 
Output: PDF report of the data?
"""

from pylatex import Document, Section, Subsection, Figure, NoEscape

def gen_title(doc, title="Bluetooth Tracking Report", author="BTU Comps"):
    """Adds a title to the document

    Args:
        doc (pylatex Document() class): Title for document
        title (str, optional): _description_. Defaults to "Bluetooth Tracking Report".
        author (str, optional): _description_. Defaults to "BTU Comps".
    """
    # Add a title
    doc.preamble.append(NoEscape(r'\title{%s}' % title))
    doc.preamble.append(NoEscape(r'\author{%s}' % author))
    doc.preamble.append(NoEscape(r'\date{\today}'))
    doc.append(NoEscape(r'\maketitle'))

def add_section(doc, sectionName="", sectionText="", imgPath="", caption="", newPage=True, imgwidth=1.1):
    """Adds a section to the document, cna include text and images

    Args:
        doc (pylatex Document() class): _description_
        sectionName (str, optional): Name of section to be added. Defaults to "".
        sectionText (str, optional): Text to appear below section. Defaults to "".
        imgPath (str, optional): Absolute path to image you want to include. Defaults to "".
        caption (str, optional): Caption that shows below image. Defaults to "".
    """
    # Add new page
    if newPage: 
        doc.append(NoEscape(r'\newpage'))

    # Add text
    if sectionName != "":
        with doc.create(Section(sectionName)):
            doc.append(sectionText)

    # Add image
    if imgPath != "":
        with doc.create(Figure(position='h!')) as img:
            img.add_image(imgPath, width=NoEscape(r'%s\textwidth ' % imgwidth))
            if caption != "":
                img.add_caption(caption)

def gen_pdf(doc, title):
    """Exports the document into a pdf

    Args:
        doc (pylatex Document() class): _description_
    """
    # set cur dir to bluetooth-tracking-ultimate
    
    doc.generate_pdf(title, clean_tex=False)  # Set clean_tex=True to remove .tex file after compilation
    print("PDF generated")

if __name__ == '__main__':
    doc = Document()
    gen_title(doc)
    add_section(doc, sectionName="Introduction", sectionText="This is the introduction to the report", imgPath="/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/charts/b4d_distance.png", caption="This is a caption")
    gen_pdf(doc)

