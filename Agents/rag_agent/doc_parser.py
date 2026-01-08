import logging
from pathlib import Path
from typing import Any, List, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem


class MedicalDocParser:
    """
    MedicalDocParser extracts structured content and figures from medical research PDFs.

    This class uses the Docling PDF pipeline to:
    - Parse PDFs into a structured document representation (pages, tables, pictures, etc.)
    - Export page renders as images (useful for debugging and traceability)
    - Export table and figure images for downstream summarization
    - Return both:
        1) The parsed Docling document object
        2) A list of image URIs/paths to be used by the image summarization pipeline

    Notes:
    - Table extraction uses TableFormerMode.ACCURATE by default for higher fidelity.
    - OCR can be enabled to recover text from scanned PDFs.
    """

    def __init__(self):
        """
        Initialize the parser and attach a module-level logger.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Medical Document Parser initialized!")

    def parse_document(
        self,
        document_path: str,
        output_dir: str,
        image_resolution_scale: float = 2.0,
        do_ocr: bool = True,
        do_tables: bool = True,
        do_formulas: bool = True,
        do_picture_desc: bool = False,
    ) -> Tuple[Any, List[str]]:
        """
        Parse a PDF document and extract structured content and images.

        Args:
            document_path: Path to the input PDF file.
            output_dir: Directory where extracted page/table/figure images will be saved.
            image_resolution_scale: Scaling factor applied to rendered images (higher yields clearer images).
            do_ocr: If True, runs OCR on scanned pages for text recovery.
            do_tables: If True, extracts table structure (rows/columns/cells).
            do_formulas: If True, enables formula enrichment (if supported by pipeline).
            do_picture_desc: If True, enables picture description generation (may increase runtime).

        Returns:
            A tuple of:
            - parsed_document: Docling structured document object
            - images: List of extracted image URIs/paths (intended for summarization)
        """
        # ------------------------------------------------------------------
        # Ensure output directory exists
        # ------------------------------------------------------------------
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Configure Docling PDF pipeline
        # ------------------------------------------------------------------
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,          # Render each PDF page as an image
            generate_picture_images=True,       # Enable figure extraction images
            images_scale=image_resolution_scale,
            do_ocr=do_ocr,
            do_table_structure=do_tables,
            do_formula_enrichment=do_formulas,
            do_picture_description=do_picture_desc,
        )

        # Prefer accuracy over speed for table structure extraction
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # ------------------------------------------------------------------
        # Convert the PDF into a structured Docling document
        # ------------------------------------------------------------------
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        conversion_res = converter.convert(document_path)

        # Use the input filename stem for consistent exported artifact naming
        doc_filename = conversion_res.input.file.stem

        # ------------------------------------------------------------------
        # Export rendered page images (useful for debugging and traceability)
        # ------------------------------------------------------------------
        for page_no, page in conversion_res.document.pages.items():
            page_image_filename = output_dir_path / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

        # ------------------------------------------------------------------
        # Export table and picture crops as PNG images
        # ------------------------------------------------------------------
        table_counter = 0
        picture_counter = 0
        image_paths: List[str] = []

        for element, _level in conversion_res.document.iterate_items():
            # Save table crops for inspection (optional downstream usage)
            if isinstance(element, TableItem):
                table_counter += 1
                table_img_path = output_dir_path / f"{doc_filename}-table-{table_counter}.png"
                with table_img_path.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")

            # Save picture crops and track their paths for downstream processing
            if isinstance(element, PictureItem):
                picture_img_path = output_dir_path / f"{doc_filename}-picture-{picture_counter}.png"
                with picture_img_path.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")

                image_paths.append(str(picture_img_path))
                picture_counter += 1

        # ------------------------------------------------------------------
        # Extract image URIs intended for summarization
        # ------------------------------------------------------------------
        # Docling may store picture references with URIs; these can be used by
        # an image summarizer (e.g., a vision model) if accessible.
        images: List[str] = []
        for picture in conversion_res.document.pictures:
            image = picture.image
            if image:
                images.append(str(image.uri))

        # Return both structured document and the image list for summarization
        return conversion_res.document, images
