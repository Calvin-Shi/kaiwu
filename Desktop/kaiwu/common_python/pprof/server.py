#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import subprocess
import tempfile
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import psutil
from typing import Optional, List, Dict
import time
from string import Template
from pathlib import Path

# --------------------------
# 常量定义
# --------------------------
# 火焰图采样配置
DEFAULT_SAMPLE_DURATION = 10  # 默认采样时长（秒）
MIN_SAMPLE_DURATION = 1  # 最小采样时长（秒）
MAX_SAMPLE_DURATION = 60  # 最大采样时长（秒）
MEMRAY_STACK_DEPTH = 15  # memray调用栈深度限制
HUMAN_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # 人类可读的时间格式

# HTTP响应配置
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_500_INTERNAL_ERROR = 500

# 路径配置（跨平台兼容）
PROJECT_ROOT = Path(__file__).parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
API_DOCS_TEMPLATE_PATH = TEMPLATE_DIR / "api_docs.html"


class FlameGraphRequestHandler(BaseHTTPRequestHandler):
    """HTTP请求处理类，支持静态资源、HTML文档和API接口"""

    def __init__(self, request, client_address, server):
        self.logger = server.logger
        super().__init__(request, client_address, server)

    def do_GET(self):
        """路由分发：处理不同路径的请求"""
        route_map = {
            "/": self._serve_api_docs_html,
            "/index.json": self._serve_api_docs_json,
            "/processes": self._serve_process_list,
            "/cpu/flamegraph": self._serve_cpu_flamegraph,
            "/memory/flamegraph": self._serve_memory_flamegraph,
        }

        # 提取核心路径（忽略查询参数）
        core_path = self.path.split("?")[0]

        # 优先处理静态资源请求
        if core_path.startswith("/static/"):
            self._serve_static_file(core_path)
            return

        # 处理其他路由
        handler = route_map.get(core_path)
        if handler:
            handler()
        else:
            self._send_error_response(
                status_code=HTTP_404_NOT_FOUND,
                error_code="not_found",
                message="请求的路径不存在（支持路径：/processes, /cpu/flamegraph, /memory/flamegraph）",
            )

    # --------------------------
    # 静态资源和HTML文档处理
    # --------------------------
    def _serve_static_file(self, path: str):
        """处理静态资源请求（安全版本）"""
        try:
            # 1. 检查路径是否以 /static/ 开头（确保是静态资源）
            if not path.startswith("/static/"):
                self.logger.warning(f"无效的静态资源路径: {path}")
                self._send_error_response(
                    status_code=HTTP_404_NOT_FOUND, error_code="invalid_static_path", message="静态资源路径必须以 /static/ 开头"
                )
                return

            # 2. 安全解析路径：只允许访问 static 目录下的文件
            # 去除 /static/ 前缀（使用切片，兼容所有Python版本）
            relative_path = path[8:]  # "/static/css/style.css" → "css/style.css"
            # 拼接路径（确保在 static 目录内）
            static_file_path = PROJECT_ROOT / "static" / relative_path

            # 3. 校验文件是否存在且在 static 目录内（双重安全检查）
            if not static_file_path.exists() or not static_file_path.is_file():
                self.logger.error(f"静态资源不存在: {static_file_path}")
                self._send_error_response(
                    status_code=HTTP_404_NOT_FOUND, error_code="static_file_not_found", message=f"静态资源不存在：{path}"
                )
                return

            # 4. 确保文件确实在 static 目录下（防止路径跳转）
            if not str(static_file_path).startswith(str(PROJECT_ROOT / "static")):
                self.logger.error(f"禁止访问静态资源外的文件: {static_file_path}")
                self._send_error_response(
                    status_code=HTTP_403_FORBIDDEN, error_code="forbidden_path", message="禁止访问该资源"
                )
                return

            # 5. 确定MIME类型
            mime_type = "text/css" if static_file_path.suffix == ".css" else "application/octet-stream"

            # 6. 读取并返回文件
            with open(static_file_path, "rb") as f:
                file_data = f.read()

            self.send_response(HTTP_200_OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(file_data)))
            self.end_headers()
            self.wfile.write(file_data)

        except Exception as e:
            self.logger.error(f"读取静态资源失败 [{path}]: {str(e)}")
            self._send_error_response(
                status_code=HTTP_500_INTERNAL_ERROR, error_code="static_file_error", message=f"静态资源读取失败：{str(e)}"
            )

    def _serve_api_docs_html(self):
        """根目录：返回可视化HTML格式API文档"""
        try:
            # 1. 获取API文档数据
            api_docs_data = self._build_api_docs_data()

            # 2. 读取HTML模板
            if not API_DOCS_TEMPLATE_PATH.exists():
                raise FileNotFoundError(f"HTML模板不存在：{API_DOCS_TEMPLATE_PATH}")

            with open(API_DOCS_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template_content = f.read()
            template = Template(template_content)

            # 3. 渲染动态内容
            rendered_data = {
                "service_name": api_docs_data["service"],
                "service_desc": api_docs_data["description"],
                "base_url": api_docs_data["base_url"],
                "endpoints_html": self._render_endpoints_to_html(api_docs_data["endpoints"]),
                "errors_html": self._render_errors_to_html(api_docs_data["common_errors"]),
                "deps_html": self._render_deps_to_html(api_docs_data["dependencies"]),
                "update_time": time.strftime(HUMAN_TIME_FORMAT, time.localtime()),
            }

            # 4. 生成最终HTML并返回
            html_content = template.substitute(rendered_data)
            html_bytes = html_content.encode("utf-8")

            self.send_response(HTTP_200_OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        except Exception as e:
            print(f"{str(e)}")
            self.logger.error(f"HTML文档生成失败: {str(e)}")
            self._send_error_response(
                status_code=HTTP_500_INTERNAL_ERROR, error_code="html_docs_error", message=f"API文档生成失败: {str(e)}"
            )

    def _serve_api_docs_json(self):
        """提供JSON格式的API文档（兼容旧接口）"""
        api_docs_data = self._build_api_docs_data()
        self._send_json_response(HTTP_200_OK, api_docs_data)

    # --------------------------
    # HTML渲染工具方法
    # --------------------------
    def _build_api_docs_data(self) -> Dict:
        """构建API文档数据结构"""
        base_url = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        return {
            "service": "多进程CPU火焰图与内存监控服务",
            "description": "基于py-spy生成CPU火焰图，基于memray生成内存快照与内存火焰图",
            "base_url": base_url,
            "endpoints": [
                {
                    "path": "/processes",
                    "method": "GET",
                    "description": "获取所有可监控的Python3进程（排除僵尸进程，按创建时间倒序）",
                    "response_format": "application/json",
                    "response_example": [
                        {
                            "pid": 1234,
                            "ppid": 987,
                            "name": "python3",
                            "status": "running",
                            "create_time": "2024-06-10 14:30:00",
                            "create_timestamp": 1718000000.123,
                            "cmd": "python3 /home/project/main.py --config dev",
                        }
                    ],
                },
                {
                    "path": "/cpu/flamegraph",
                    "method": "GET",
                    "description": "为指定Python3进程生成SVG格式CPU火焰图",
                    "parameters": [
                        {"name": "pid", "type": "integer", "required": True, "description": "目标进程ID（从/processes接口获取）"},
                        {
                            "name": "duration",
                            "type": "integer",
                            "required": False,
                            "default": DEFAULT_SAMPLE_DURATION,
                            "description": f"采样时长（秒），范围{MIN_SAMPLE_DURATION}-{MAX_SAMPLE_DURATION}",
                        },
                    ],
                    "request_examples": [
                        f"{base_url}/cpu/flamegraph?pid=1234",
                        f"{base_url}/cpu/flamegraph?pid=1234&duration=10",
                    ],
                    "response_format": "image/svg+xml",
                },
                {
                    "path": "/memory/flamegraph",
                    "method": "GET",
                    "description": "为指定Python3进程生成HTML格式内存火焰图（按内存分配占比展示）",
                    "parameters": [
                        {"name": "pid", "type": "integer", "required": True, "description": "目标进程ID（从/processes接口获取）"},
                        {
                            "name": "duration",
                            "type": "integer",
                            "required": False,
                            "default": DEFAULT_SAMPLE_DURATION,
                            "description": f"采样时长（秒），范围{MIN_SAMPLE_DURATION}-{MAX_SAMPLE_DURATION}",
                        },
                    ],
                    "request_examples": [
                        f"{base_url}/memory/flamegraph?pid=1234",
                        f"{base_url}/memory/flamegraph?pid=1234&duration=8",
                    ],
                    "response_format": "text/html",
                },
            ],
            "common_errors": [
                {"code": "not_found", "message": "请求路径不存在"},
                {"code": "missing_pid", "message": "缺少必填参数pid（进程ID）"},
                {"code": "invalid_pid", "message": "pid必须为整数"},
                {"code": "process_not_found", "message": "指定进程不存在或已退出"},
                {"code": "cpu_spy_error", "message": "py-spy CPU采样失败（权限不足、进程无响应等）"},
                {"code": "memray_flame_error", "message": "memray内存火焰图采样失败"},
                {"code": "static_file_not_found", "message": "静态资源（如CSS）不存在"},
                {"code": "html_docs_error", "message": "HTML格式API文档生成失败"},
                {"code": "internal_error", "message": "服务内部错误"},
            ],
            "dependencies": [
                "py-spy >= 0.3.12",
                "memray >= 1.11.0",
                "psutil >= 5.9.0",
                "Python >= 3.7",
            ],
        }

    def _render_endpoints_to_html(self, endpoints: List[Dict]) -> str:
        """将API端点列表渲染为HTML"""
        html_parts = []
        for endpoint in endpoints:
            # 构建路径部分（如果是/processes则添加链接）
            if endpoint["path"] == "/processes":
                path_html = f'<a href="{endpoint["path"]}" class="path">{endpoint["path"]}</a>'
            else:
                path_html = f'<span class="path">{endpoint["path"]}</span>'

            # 方法样式（GET/POST等）
            method_html = f'<span class="method">{endpoint["method"]}</span>'

            # 描述部分
            desc_html = f'<p class="description">{endpoint["description"]}</p>'

            # 参数表格
            params_html = ""
            if "parameters" in endpoint:
                params_html = '<h4 class="param-title">参数说明：</h4>'
                params_html += '<table class="params-table"><thead><tr><th>参数名</th><th>类型</th><th>是否必填</th><th>默认值</th><th>说明</th></tr></thead><tbody>'
                for param in endpoint["parameters"]:
                    params_html += f'<tr><td>{param["name"]}</td><td>{param["type"]}</td><td>{"是" if param.get("required") else "否"}</td><td>{param.get("default", "-")}</td><td>{param["description"]}</td></tr>'
                params_html += "</tbody></table>"

            # 请求示例
            examples_html = ""
            if "request_examples" in endpoint:
                examples_html = '<h4 class="example-title">请求示例：</h4><ul class="examples-list">'
                for example in endpoint["request_examples"]:
                    examples_html += f"<li>{example}</li>"
                examples_html += "</ul>"

            # 组合单个接口卡片
            html_parts.append(
                f"""
                <div class="endpoint-card">
                    <div class="path-method">
                        {method_html}
                        {path_html}
                    </div>
                    {desc_html}
                    {params_html}
                    {examples_html}
                    <p class="response-format"><strong>响应格式：</strong>{endpoint["response_format"]}</p>
                </div>
            """
            )

        return "".join(html_parts)

    def _render_errors_to_html(self, errors: List[Dict]) -> str:
        """将错误码列表转换为HTML表格片段"""
        error_rows = []
        for error in errors:
            error_rows.append(
                f"""
                <tr>
                    <td>{error['code']}</td>
                    <td>{error['message']}</td>
                </tr>
            """
            )
        return "".join(error_rows)

    def _render_deps_to_html(self, deps: List[str]) -> str:
        """将依赖列表转换为HTML列表片段"""
        dep_items = []
        for dep in deps:
            dep_items.append(f"<li>{dep}</li>")
        return "".join(dep_items)

    # --------------------------
    # 核心业务处理方法
    # --------------------------
    def _serve_process_list(self):
        """返回HTML格式的进程列表，包含火焰图链接"""
        try:
            # 获取进程列表数据
            processes = self._get_python_processes()

            # 构建基础URL
            base_url = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"

            # 渲染进程列表HTML
            processes_html = []
            for proc in processes:
                # 生成火焰图相对路径链接（默认duration=10s）
                cpu_url = f"/cpu/flamegraph?pid={proc['pid']}&duration=10"
                mem_url = f"/memory/flamegraph?pid={proc['pid']}&duration=10"

                processes_html.append(
                    f"""
                <tr>
                    <td>{proc['pid']}</td>
                    <td>{proc['ppid']}</td>
                    <td>{proc['name']}</td>
                    <td>{proc['status']}</td>
                    <td>{proc['create_time']}</td>
                    <td>{proc['cmd']}</td>
                    <td class="operation-links">
                        <a href="{cpu_url}" target="_blank">CPU火焰图</a>
                        <a href="{mem_url}" target="_blank">内存火焰图</a>
                    </td>
                </tr>
                """
                )

            # 读取HTML模板
            processes_template_path = TEMPLATE_DIR / "processes.html"
            if not processes_template_path.exists():
                raise FileNotFoundError(f"进程列表模板不存在：{processes_template_path}")

            with open(processes_template_path, "r", encoding="utf-8") as f:
                template_content = f.read()
            template = Template(template_content)

            # 渲染模板
            rendered_data = {
                "base_url": base_url,
                "processes_html": "\n".join(processes_html),
                "update_time": time.strftime(HUMAN_TIME_FORMAT, time.localtime()),
            }

            html_content = template.substitute(rendered_data)
            html_bytes = html_content.encode("utf-8")

            self.send_response(HTTP_200_OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        except Exception as e:
            self.logger.error(f"获取进程列表失败: {str(e)}")
            self._send_error_response(
                status_code=HTTP_500_INTERNAL_ERROR, error_code="process_list_error", message=f"获取进程列表失败：{str(e)}"
            )

    def _get_python_processes(self) -> List[Dict]:
        """获取所有Python3进程信息（工具方法）"""
        python_processes = []
        for proc in psutil.process_iter(["pid", "ppid", "name", "status", "create_time", "cmdline"]):
            try:
                # 过滤出Python进程且非僵尸进程
                if "python" in proc.info["name"].lower() and proc.info["status"] != "zombie":
                    # 格式化命令行
                    cmd = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else "N/A"

                    # 格式化创建时间
                    create_time = time.strftime(HUMAN_TIME_FORMAT, time.localtime(proc.info["create_time"]))

                    python_processes.append(
                        {
                            "pid": proc.info["pid"],
                            "ppid": proc.info["ppid"],
                            "name": proc.info["name"],
                            "status": proc.info["status"],
                            "create_time": create_time,
                            "create_timestamp": proc.info["create_time"],
                            "cmd": cmd,
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # 按创建时间倒序排列（最新的在前面）
        return sorted(python_processes, key=lambda x: x["create_timestamp"], reverse=True)

    def _serve_cpu_flamegraph(self):
        """处理/cpu/flamegraph请求：为指定PID生成SVG格式CPU火焰图"""
        # 1. 解析并校验参数
        params = self._parse_query_params()
        pid, duration = self._validate_flamegraph_params(params)
        if not pid:
            return

        # 2. 检查进程是否存在
        if not psutil.pid_exists(pid):
            return self._send_error_response(
                status_code=HTTP_404_NOT_FOUND, error_code="process_not_found", message=f"进程不存在: {pid}"
            )

        # 3. 生成CPU火焰图
        self.logger.info(f"开始CPU采样 [PID:{pid}]，时长 {duration} 秒")
        svg_temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=".svg", delete=False) as svg_file:
                svg_temp_path = svg_file.name
                self._run_py_spy(pid, duration, svg_temp_path)

            # 返回SVG文件
            with open(svg_temp_path, "rb") as f:
                svg_data = f.read()
            self._send_svg_response(pid, duration, svg_data, is_memory=False)

        except subprocess.CalledProcessError as e:
            error_msg = f"py-spy CPU采样失败: {e.stderr.strip()}"
            self.logger.error(error_msg)
            self._send_error_response(HTTP_500_INTERNAL_ERROR, "cpu_spy_error", error_msg)
        except Exception as e:
            error_msg = f"CPU火焰图生成失败: {str(e)}"
            self.logger.error(error_msg)
            self._send_error_response(HTTP_500_INTERNAL_ERROR, "cpu_flamegraph_error", error_msg)
        finally:
            self._cleanup_temp_file(svg_temp_path)

    def _serve_memory_flamegraph(self):
        """处理/memory/flamegraph请求：生成HTML格式内存火焰图"""
        # 1. 解析并校验参数
        params = self._parse_query_params()
        pid, duration = self._validate_flamegraph_params(params)
        if not pid:
            return

        # 2. 检查进程是否存在
        if not psutil.pid_exists(pid):
            return self._send_error_response(
                status_code=HTTP_404_NOT_FOUND, error_code="process_not_found", message=f"进程不存在: {pid}"
            )

        # 3. 生成内存火焰图（HTML格式）
        self.logger.info(f"开始内存采样 [PID:{pid}]，时长 {duration} 秒")
        html_temp_path = None
        memray_temp_path = None
        try:
            # 3.1 第一步：生成内存快照文件（.bin）
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as mem_file:
                memray_temp_path = mem_file.name
                self._run_memray_attach(pid, duration, memray_temp_path)

            # 3.2 第二步：从快照生成HTML火焰图
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=".html", delete=False) as html_file:
                html_temp_path = html_file.name
                self._run_memray_flamegraph(memray_temp_path, html_temp_path)

            # 3.3 读取HTML内容
            with open(html_temp_path, "rb") as f:
                html_data = f.read()

            self.logger.info(f"内存火焰图生成完成 [PID:{pid}]")
            self._send_html_response(pid, duration, html_data)

        except subprocess.CalledProcessError as e:
            error_msg = f"memray火焰图采样失败: {e.stderr.strip()}"
            self.logger.error(error_msg)
            self._send_error_response(HTTP_500_INTERNAL_ERROR, "memray_flame_error", error_msg)
        except Exception as e:
            error_msg = f"内存火焰图生成失败: {str(e)}"
            self.logger.error(error_msg)
            self._send_error_response(HTTP_500_INTERNAL_ERROR, "memray_html_error", error_msg)
        finally:
            # 清理临时文件
            self._cleanup_temp_file(html_temp_path)
            self._cleanup_temp_file(memray_temp_path)

    # --------------------------
    # 工具方法
    # --------------------------
    def _get_filtered_processes(self) -> List[Dict]:
        """获取过滤后的Python3进程：排除僵尸进程，保留关键信息"""
        target_procs = []
        for proc in psutil.process_iter(["pid", "ppid", "name", "status", "cmdline", "create_time"]):
            try:
                proc_info = proc.info
                # 过滤1：进程名以python3开头
                if not proc_info["name"].startswith("python3"):
                    continue
                # 过滤2：排除僵尸进程
                if proc_info["status"] == psutil.STATUS_ZOMBIE:
                    continue

                # 转换创建时间为人类可读格式
                create_timestamp = proc_info["create_time"]
                human_create_time = time.strftime(HUMAN_TIME_FORMAT, time.localtime(create_timestamp))

                # 构造进程信息
                target_procs.append(
                    {
                        "pid": proc_info["pid"],
                        "ppid": proc_info["ppid"],
                        "name": proc_info["name"],
                        "status": proc_info["status"],
                        "create_time": human_create_time,
                        "create_timestamp": create_timestamp,
                        "cmd": " ".join(proc_info["cmdline"] or [])[:150],
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
        # 按创建时间倒序（最新进程在前）
        return sorted(target_procs, key=lambda x: x["create_timestamp"], reverse=True)

    def _parse_query_params(self) -> Dict[str, str]:
        """解析URL查询参数，返回字典格式"""
        params = {}
        if "?" not in self.path:
            return params

        query_str = self.path.split("?")[1]
        for param in query_str.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                params[key.strip()] = value.strip()
        return params

    def _validate_pid_param(self, params: Dict[str, str]) -> Optional[int]:
        """单独校验PID参数"""
        if "pid" not in params:
            self._send_error_response(
                status_code=HTTP_400_BAD_REQUEST, error_code="missing_pid", message="参数错误: 缺少必填参数 pid（进程ID）"
            )
            return None

        try:
            return int(params["pid"])
        except ValueError:
            self._send_error_response(
                status_code=HTTP_400_BAD_REQUEST, error_code="invalid_pid", message="参数错误: pid必须为整数"
            )
            return None

    def _validate_flamegraph_params(self, params: Dict[str, str]) -> (Optional[int], int):
        """校验火焰图参数：返回(pid, duration)或(None, 0)"""
        # 先校验PID
        pid = self._validate_pid_param(params)
        if not pid:
            return None, 0

        # 校验duration
        try:
            duration = int(params.get("duration", DEFAULT_SAMPLE_DURATION))
            duration = max(MIN_SAMPLE_DURATION, min(duration, MAX_SAMPLE_DURATION))
        except ValueError:
            self.logger.warning(f"duration参数无效（{params.get('duration')}），使用默认值 {DEFAULT_SAMPLE_DURATION} 秒")
            duration = DEFAULT_SAMPLE_DURATION

        return pid, duration

    def _run_py_spy(self, pid: int, duration: int, output_path: str):
        """执行py-spy命令（CPU火焰图）"""
        spy_cmd = [
            "py-spy",
            "record",
            "--pid",
            str(pid),
            "--format",
            "flamegraph",
            "--duration",
            str(duration),
            "--output",
            output_path,
        ]
        subprocess.run(spy_cmd, check=True, capture_output=True, text=True)

    def _run_memray_attach(self, pid: int, duration: int, output_path: str):
        """执行memray attach命令（采集内存快照）"""
        memray_cmd = [
            "memray",
            "attach",
            str(pid),
            "--output",
            output_path,
            "--duration",
            str(duration),
            "--force",
        ]
        subprocess.run(memray_cmd, check=True, capture_output=True, text=True)

    def _run_memray_flamegraph(self, snapshot_path: str, output_path: str):
        """执行memray flamegraph命令（从快照文件生成火焰图）"""
        memray_cmd = [
            "memray",
            "flamegraph",
            snapshot_path,
            "--output",
            output_path,
            "--force",
        ]
        subprocess.run(memray_cmd, check=True, capture_output=True, text=True)

    def _cleanup_temp_file(self, file_path: Optional[str]):
        """清理临时文件"""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                self.logger.warning(f"清理临时文件失败 [{file_path}]: {str(e)}")

    # --------------------------
    # HTTP响应工具方法
    # --------------------------
    def _send_json_response(self, status_code: int, data: Dict):
        """发送JSON格式响应"""
        response_body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _send_svg_response(self, pid: int, duration: int, svg_data: bytes, is_memory: bool):
        """发送SVG格式响应"""
        flame_type = "memory" if is_memory else "cpu"
        filename = f"{flame_type}_flamegraph_pid{pid}_{duration}s.svg"
        self.send_response(HTTP_200_OK)
        self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("Content-Length", str(len(svg_data)))
        self.end_headers()
        self.wfile.write(svg_data)

    def _send_html_response(self, pid: int, duration: int, html_data: bytes):
        """发送HTML格式响应"""
        filename = f"memory_flamegraph_pid{pid}_{duration}s.html"
        self.send_response(HTTP_200_OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("Content-Length", str(len(html_data)))
        self.end_headers()
        self.wfile.write(html_data)

    def _send_error_response(self, status_code: int, error_code: str, message: str):
        """发送统一格式的错误响应"""
        error_data = {"error": error_code, "message": message, "status_code": status_code}
        self._send_json_response(status_code, error_data)


def start_flamegraph_server(host="0.0.0.0", port=8080, logger=None):
    """启动CPU火焰图与内存监控服务"""
    # 1. 校验外部依赖
    if logger is None:
        raise ValueError("必须传入logger参数（logging.Logger实例），用于服务日志记录")

    # 检查py-spy是否安装
    try:
        subprocess.run(["py-spy", "--version"], capture_output=True, check=True, text=True)
    except Exception as e:
        error_msg = f"依赖检查失败：未安装py-spy（请执行 `pip install py-spy` 安装），错误详情: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # 检查memray是否安装
    try:
        subprocess.run(["memray", "--version"], capture_output=True, check=True, text=True)
    except Exception as e:
        error_msg = f"依赖检查失败：未安装memray（请执行 `pip install memray` 安装），错误详情: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # 2. 确保模板和静态资源目录存在
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)

    # 3. 自定义HTTPServer类：传递logger到RequestHandler
    class LoggerHTTPServer(HTTPServer):
        def __init__(self, server_address, RequestHandlerClass):
            super().__init__(server_address, RequestHandlerClass)
            self.logger = logger  # 存储外部logger，供Handler访问

    # 4. 启动HTTP服务
    try:
        server = LoggerHTTPServer((host, port), FlameGraphRequestHandler)
        logger.info(f"CPU火焰图与内存监控服务启动成功: http://{host}:{port}")
        logger.info(f"  - 可视化API文档: http://{host}:{port}/")
        logger.info(f"  - 进程列表: http://{host}:{port}/processes")
        logger.info(f"  - CPU火焰图: http://{host}:{port}/cpu/flamegraph?pid=xxx&duration=10")
        logger.info(f"  - 内存火焰图: http://{host}:{port}/memory/flamegraph?pid=xxx&duration=10")

        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        return server
    except Exception as e:
        error_msg = f"服务启动失败: {str(e)}（可能是端口被占用，建议更换port参数）"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
