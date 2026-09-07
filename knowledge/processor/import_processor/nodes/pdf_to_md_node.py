import contextlib
import io
import json
import time
from pathlib import Path
from typing import Tuple

import click
from mineru.cli.client import main as mineru_main

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import PdfConversionError, StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口

        :param state: 导入图谱的状态字典，必须包含 import_file_path，可选包含 file_dir
        :return: 更新后的 state 字典，新增 md_path 字段
        """
        import_file_path_obj, file_dir_obj = self._validate_state(state)

        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError(message="MinerU解析PDF失败", node_name=self.name)

        md_path = self._get_md_path(import_file_path_obj, file_dir_obj)

        state['md_path'] = md_path

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        校验并从 state 中提取导入文件路径与输出目录

        :param state: 导入图谱节点状态
        :return: (导入文件路径对象, 输出目录对象)
        """
        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        import_file_path = state.get('import_file_path', '')

        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str,
                                  message="解析文件的路径不存在")

        file_dir = state.get('file_dir', '')

        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_obj = Path(file_dir)

        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str,
                                  message="输出目录不存在")

        self.logger.info(f"解析的文件路径{import_file_path}")
        self.logger.info(f"输出的文件目录{file_dir}")

        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path,
                              file_dir_obj: Path) -> int:
        """
        调用 mineru CLI 将 PDF 解析为 Markdown

        :param import_file_path_obj: 解析文件的 path 路径
        :param file_dir_obj: 解析后的文件输出目录
        :return: 退出码，0 表示成功，非 0 表示失败
        """
        start_time = time.time()

        args = [
            "-p", str(import_file_path_obj),
            "-o", str(file_dir_obj),
            "-b", "pipeline",
            "-m", "txt",
            "-l", "ch",
        ]

        exit_code = 0
        output_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buf), contextlib.redirect_stderr(output_buf):
                mineru_main(args=args, standalone_mode=False)
        except click.exceptions.Exit as e:
            exit_code = e.exit_code
        except Exception as e:
            self.logger.error(f"MinerU解析异常: {e}")
            exit_code = 1

        for line in output_buf.getvalue().splitlines():
            self.logger.info(f"MinerU: {line}")

        end_time = time.time()
        if exit_code == 0:
            self.logger.info(f"MinerU解析PDF成功 耗时:{end_time - start_time:.2f}s")
        else:
            self.logger.error(f"MinerU解析PDF失败 exit_code={exit_code}")

        return exit_code

    @staticmethod
    def _get_md_path(import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        根据输入路径和输出目录，拼出解析后 md 文件的绝对路径

        :param import_file_path_obj: 原始 PDF 文件路径
        :param file_dir_obj: 输出根目录
        :return: md 文件的绝对路径字符串
        """
        file_name = import_file_path_obj.stem
        return str(file_dir_obj / file_name / "txt" / f"{file_name}.md")


if __name__ == '__main__':
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    init_state = {
        "import_file_path": r"D:\PyCharm项目\shopkeeper_brain_1\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        "file_dir": r"D:\PyCharm项目\shopkeeper_brain_1\knowledge\processor\import_processor\temp_dir"
    }

    result = pdf_to_md_node.process(init_state)

    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)