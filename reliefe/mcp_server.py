"""Optional local stdio MCP server. Install with ``reliefe[mcp]`` (Python 3.10+)."""

import sys
from typing import List, Optional, Union

from .ranking import rank_features


def create_server():
    """Construct the server without starting it or configuring global logging."""
    if sys.version_info < (3, 10):
        raise SystemExit("The MCP server requires Python 3.10 or newer.")
    try:
        from typing import Annotated
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
        from pydantic import BaseModel, Field, StrictInt, StrictStr
    except ImportError as error:
        raise SystemExit('Install the MCP extra: pip install "reliefe[mcp]"') from error

    class Feature(BaseModel):
        index: int
        score: float
        name: Optional[str]

    class RankingResult(BaseModel):
        scores: List[float]
        ranking: List[Feature]
        n_samples: int
        n_features: int
        num_iter: int
        k: int
        method: str
        normalization: str

    server = FastMCP("ReliefE", instructions=rank_features.__doc__)
    Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
    Row = Annotated[List[Number], Field(min_length=1, max_length=1024)]

    @server.tool(name="rank_features", description=rank_features.__doc__, annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True,
        openWorldHint=False))
    def rank(x: Annotated[List[Row], Field(min_length=4, max_length=512)],
             y: Annotated[List[Union[StrictInt, StrictStr]], Field(min_length=4, max_length=512)],
             feature_names: Optional[List[StrictStr]] = None,
             top_k: Optional[Annotated[StrictInt, Field(ge=1, le=1024)]] = None,
             num_iter: Annotated[StrictInt, Field(ge=1, le=512)] = 128,
             k: Annotated[StrictInt, Field(ge=1, le=511)] = 1) -> RankingResult:
        return RankingResult(**rank_features(x, y, feature_names, top_k, num_iter, k))

    return server


def main():
    """Run stdio only: no network listener, arbitrary file reads, or file writes."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
