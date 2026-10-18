import importlib.util
import os
from datetime import timedelta
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

from reliefe import mcp_server


SDK_AVAILABLE = sys.version_info >= (3, 10) and importlib.util.find_spec("mcp") is not None


class OptionalServerTests(unittest.TestCase):
    def test_unsupported_python_and_missing_extra(self):
        with patch.object(sys, "version_info", (3, 9)):
            with self.assertRaisesRegex(SystemExit, "Python 3.10"):
                mcp_server.create_server()
        with patch.dict(sys.modules, {"mcp.server.fastmcp": None}), \
             patch.object(sys, "version_info", (3, 10)):
            with self.assertRaisesRegex(SystemExit, "pip install"):
                mcp_server.create_server()

    @unittest.skipUnless(SDK_AVAILABLE, "MCP extra requires Python 3.10+")
    def test_entrypoints(self):
        with patch("mcp.server.fastmcp.FastMCP.run") as run:
            mcp_server.main()
            run.assert_called_once_with(transport="stdio")
        with patch("mcp.server.fastmcp.FastMCP.run") as run:
            with patch.dict(sys.modules):
                sys.modules.pop("reliefe.mcp_server", None)
                runpy.run_module("reliefe.mcp_server", run_name="__main__")
            run.assert_called_once_with(transport="stdio")


@unittest.skipUnless(SDK_AVAILABLE, "MCP extra requires Python 3.10+")
class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_results_and_recoverable_errors(self):
        server = mcp_server.create_server()
        tools = await server.list_tools()
        self.assertEqual([tool.name for tool in tools], ["rank_features"])
        tool = tools[0]
        self.assertIn("classification", tool.description)
        self.assertTrue(tool.annotations.readOnlyHint)
        self.assertFalse(tool.annotations.openWorldHint)
        self.assertIn("ranking", tool.outputSchema["properties"])
        self.assertEqual(tool.inputSchema["properties"]["x"]["maxItems"], 512)
        for args in ({"x": [[0.], [1.], [2.], [3.]], "y": [0, 0, 0, 1]},
                     {"x": [[True]] * 4, "y": [0, 0, 1, 1]},
                     {"x": [[0.]] * 4, "y": [0, 0, 1, 1], "k": True}):
            with self.assertRaises(Exception):
                await server.call_tool("rank_features", args)
        result = await server.call_tool("rank_features", {
            "x": [[0., 0., 1.], [0., 1., 1.], [1., 0., 0.], [1., 1., 0.]],
            "y": [0, 0, 1, 1], "top_k": 2})
        # FastMCP's direct API returns text blocks and structured data.
        self.assertEqual(result[1]["scores"], [1., -1., 1.])

    async def test_real_stdio_round_trip(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "reliefe.mcp_server"], cwd=str(Path(__file__).resolve().parents[1]),
            env={**os.environ, "NUMBA_DISABLE_JIT": "1"})
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=20)) as session:
                await session.initialize()
                self.assertEqual([tool.name for tool in (await session.list_tools()).tools], ["rank_features"])
                error = await session.call_tool("rank_features", {"x": [[0.]] * 4, "y": [0, 0, 0, 1]})
                self.assertTrue(error.isError)
                result = await session.call_tool("rank_features", {
                    "x": [[0., 0., 1.], [0., 1., 1.], [1., 0., 0.], [1., 1., 0.]],
                    "y": ["a", "a", "b", "b"], "num_iter": 4, "k": 1})
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent["scores"], [1., -1., 1.])


if __name__ == "__main__":
    unittest.main()
