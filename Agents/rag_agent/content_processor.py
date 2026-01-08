import re
import logging
from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ContentProcessor:
    """
    ContentProcessor prepares parsed documents for Retrieval-Augmented Generation (RAG).

    Responsibilities:
    1) Summarize embedded images (figures, charts, screenshots) using a vision-capable LLM.
    2) Inject image summaries into the parsed document at placeholder positions.
    3) Create semantically coherent text chunks suitable for embedding and retrieval.

    Notes:
    - This class assumes the upstream parser provides an export_to_markdown method that
      supports placeholders for images and page breaks.
    - Chunking is performed in two phases:
        (a) deterministic pre-split by markdown section boundaries
        (b) LLM-guided grouping into 256–512 word semantic sections
    """

    def __init__(self, config):
        """
        Initialize processor with models configured for summarization and chunking.

        Args:
            config: Project configuration object providing:
                - config.rag.summarizer_model (temperature ~0.5)
                - config.rag.chunker_model (temperature ~0.0 for deterministic splitting)
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model
        self.chunker_model = config.rag.chunker_model

    def summarize_images(self, images: List[str]) -> List[str]:
        """
        Summarize a list of images using the configured summarizer model.

        Each image is expected to be a URL or a data-url (base64) that the model can read.
        If an image is not informative (e.g., UI icons/buttons), the model is instructed
        to return the literal string "non-informative", which downstream steps can drop.

        Args:
            images: List of image URLs/data-urls.

        Returns:
            A list of summaries aligned with the input list.
            If summarization fails for an image, a placeholder summary is returned.
        """
        prompt_template = (
            "Describe the image in detail while keeping it concise and to the point. "
            "For context, the image is part of a medical research paper or medical report, "
            "often demonstrating AI/ML/DL for disease diagnosis. "
            "Be specific about graphs (e.g., bar plots) if present. "
            "Only summarize what is present in the image without adding extra commentary. "
            "If the image is not relevant (e.g., UI buttons/icons), return 'non-informative'."
        )

        # ChatPromptTemplate supports multi-modal messages (text + image)
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {"type": "image_url", "image_url": {"url": "{image}"}},
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()

        results: List[str] = []
        for image in images:
            try:
                summary = summary_chain.invoke({"image": image})
                results.append(summary)
            except Exception as e:
                # Log and continue so the pipeline remains robust
                self.logger.warning(f"Failed to summarize image: {image}. Error: {str(e)}")
                results.append("no image summary")

        return results

    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Export parsed document content to markdown and replace image placeholders with summaries.

        Args:
            parsed_document: Parsed document object from doc_parser with export_to_markdown support.
            image_summaries: Summaries returned from summarize_images, aligned by occurrence order.

        Returns:
            Markdown text where image placeholders are replaced by short summaries.
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"

        # Export parsed document to markdown, preserving placeholders
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
            image_placeholder=IMAGE_PLACEHOLDER
        )

        # Replace placeholders sequentially with summaries
        formatted_document = self._replace_occurrences(
            formatted_parsed_document,
            IMAGE_PLACEHOLDER,
            image_summaries
        )

        return formatted_document

    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace placeholders in text sequentially with provided replacements.

        Behavior:
        - Replaces one occurrence at a time, in order.
        - If a replacement is "non-informative", the placeholder is removed.
        - Stops gracefully when no more placeholders exist.

        Args:
            text: Input text containing placeholders.
            target: Placeholder token to replace.
            replacements: Replacement strings, one per placeholder occurrence.

        Returns:
            Text with placeholders replaced or removed.
        """
        result = text

        for counter, replacement in enumerate(replacements):
            if target not in result:
                # No more placeholders to replace
                break

            # Remove unhelpful image summaries to keep context clean
            if replacement.lower() == "non-informative":
                result = result.replace(target, "", 1)
                continue

            # Prefix a stable identifier so images can be referenced later if needed
            result = result.replace(
                target,
                f"picture_counter_{counter} {replacement}",
                1
            )

        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Create semantically coherent chunks for embedding and retrieval.

        Two-step approach:
        1) Pre-split by markdown section boundaries (simple heuristic).
        2) Use an LLM to suggest where to merge/split chunks to achieve:
           - thematic cohesion
           - target chunk size (256–512 words)
           - ordered chunk grouping

        Args:
            formatted_document: Document text where images are already summarized/inserted.

        Returns:
            List of final chunk strings (semantic sections).
        """
        # Pre-split by markdown section headers. This keeps headings grouped with content.
        SPLIT_PATTERN = "\n#"
        raw_chunks = formatted_document.split(SPLIT_PATTERN)

        # Wrap each initial chunk with markers so the LLM can reference boundaries
        chunked_text = ""
        for i, chunk in enumerate(raw_chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"

        # LLM-guided chunk grouping instruction prompt
        CHUNKING_PROMPT = """
You are an assistant specialized in splitting text into semantically consistent sections.

<document>
{document_text}
</document>

<instructions>
1. The text is divided into chunks marked with <|start_chunk_X|> and <|end_chunk_X|>.
2. Decide where splits should occur so consecutive chunks of the same theme stay together.
3. Each final section must be between 256 and 512 words.
4. If chunk 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
5. Output split points in ascending order.
6. Output format must be: split_after: 3, 5
</instructions>

Respond only with the chunk IDs where a split should occur.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
""".strip()

        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)

        # Chunker model returns a string like: "split_after: 2, 5"
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content

        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)

    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        Convert LLM split suggestions into final grouped document sections.

        Args:
            chunked_text: Marker-wrapped chunks (<|start_chunk_X|> ... <|end_chunk_X|>).
            llm_response: LLM output indicating split points (e.g., "split_after: 2, 5").

        Returns:
            List of final semantic sections as plain text.
        """
        # Parse split points from the LLM response
        split_after: List[int] = []
        if "split_after:" in llm_response:
            split_points_str = llm_response.split("split_after:", 1)[1].strip()
            split_after = [int(x.strip()) for x in split_points_str.replace(",", " ").split()]

        # Defensive fallback if the model output is malformed
        if not split_after:
            return [chunked_text]

        # Extract chunk contents by matching the marker pairs
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # Group chunks into sections according to split points
        sections: List[str] = []
        current_section: List[str] = []

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)

            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = []

        # Append remaining content as the final section
        if current_section:
            sections.append("".join(current_section).strip())

        return sections
