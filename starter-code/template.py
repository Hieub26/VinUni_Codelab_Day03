"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> str:
        """Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)"""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "answer": response.text,
                    "status": "success",
                    "tool_calls": [],
                    "trace": [
                        {"step": "init", "user_input": user_input, "final_answer": response.text}
                    ]
                }
            except Exception as e:
                return {
                    "answer": f"Error: {str(e)}",
                    "status": "fail",
                    "tool_calls": [],
                    "trace": []
                }
        # Fallback khi không có API key
        return {
            "answer": "Không có API key để gọi LLM.",
            "status": "fail",
            "tool_calls": [],
            "trace": []
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.trace = []
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def _call_llm(self, messages: list) -> str:
        """Gọi Gemini API với danh sách messages và trả về text response."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Ghép tất cả messages thành 1 prompt duy nhất
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[System]\n{content}\n\n"
            elif role == "user":
                prompt += f"[User]\n{content}\n\n"
            elif role == "assistant":
                prompt += f"[Assistant]\n{content}\n\n"
        response = model.generate_content(prompt)
        return response.text

    def _parse_action(self, text: str) -> dict | None:
        """Parse Action JSON từ response text của LLM.
        
        Tìm dòng bắt đầu bằng 'Action:' và parse JSON phía sau.
        Trả về dict {"name": ..., "args": ...} hoặc None nếu không tìm thấy.
        """
        # Tìm dòng Action trong response
        action_match = re.search(r'Action:\s*(\{.*?\})\s*$', text, re.MULTILINE | re.DOTALL)
        if not action_match:
            # Thử tìm JSON block sau "Action:"
            action_match = re.search(r'Action:\s*```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not action_match:
            return None

        json_str = action_match.group(1).strip()
        try:
            action = json.loads(json_str)
            if "name" in action:
                return action
        except json.JSONDecodeError:
            pass
        return None

    def _parse_final_answer(self, text: str) -> str | None:
        """Parse Final Answer từ response text của LLM.
        
        Tìm dòng bắt đầu bằng 'Final Answer:' và trả về nội dung phía sau.
        """
        match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, action: dict) -> str:
        """Thực thi tool từ TOOL_MAP dựa trên action dict.
        
        Xử lý các trap phổ biến:
        - Trap 1: .strip().lower() tên tool để tránh KeyError
        - Trap 2: try/except cho JSON parse errors
        """
        tool_name = action["name"].strip().lower()
        args = action.get("args", {})

        if tool_name not in TOOL_MAP:
            return json.dumps({"error": f"Tool '{tool_name}' không tồn tại trong TOOL_MAP."}, ensure_ascii=False)

        try:
            tool_fn = TOOL_MAP[tool_name]
            result = tool_fn(**args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"Lỗi khi gọi tool '{tool_name}': {str(e)}"}, ensure_ascii=False)

    def run(self, user_input: str) -> dict:
        """Chạy ReAct Loop: Thought → Action → Observation → ... → Final Answer
        
        Returns:
            dict với keys: status, answer, iterations, trace
        """
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        iteration = 0
        tool_definitions_str = json.dumps(TOOL_DEFINITIONS, indent=2, ensure_ascii=False)
        system_prompt = SYSTEM_PROMPT.format(tools=tool_definitions_str)

        # Lịch sử conversation để gửi cho LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            try:
                llm_response = self._call_llm(messages)
            except Exception as e:
                return {
                    "status": "error",
                    "answer": f"Lỗi khi gọi LLM: {str(e)}",
                    "iterations": iteration,
                    "trace": self.trace
                }

            # Parse Final Answer trước — nếu có thì kết thúc
            final_answer = self._parse_final_answer(llm_response)

            # Parse Action
            action = self._parse_action(llm_response)

            # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
            if action:
                # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
                observation = self._execute_tool(action)

                # Ghi trace
                self.trace.append({
                    "step": iteration,
                    "thought": llm_response.split("Action:")[0].replace("Thought:", "").strip(),
                    "action": action,
                    "observation": observation
                })

                # Append vào conversation history để LLM biết kết quả
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            elif final_answer:
                # Không có Action, có Final Answer → kết thúc
                self.trace.append({
                    "step": iteration,
                    "thought": llm_response.split("Final Answer:")[0].replace("Thought:", "").strip(),
                    "final_answer": final_answer
                })

                return {
                    "status": "completed",
                    "answer": final_answer,
                    "iterations": iteration,
                    "trace": self.trace
                }
            else:
                # LLM không trả về Action hay Final Answer → ghi nhận và thử lại
                self.trace.append({
                    "step": iteration,
                    "raw_response": llm_response,
                    "note": "No Action or Final Answer detected"
                })
                # Gửi lại yêu cầu format đúng
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": "Observation: Invalid format. Hãy trả lời theo đúng format: "
                               "Thought: ... Action: {\"name\": ..., \"args\": {...}} hoặc Final Answer: ..."
                })

        # Milestone 4: Safeguard — đã vượt max_iterations
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "iterations": iteration,
            "trace": self.trace
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
