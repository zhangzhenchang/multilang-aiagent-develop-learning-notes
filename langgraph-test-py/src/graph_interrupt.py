"""graph_interrupt.py - interrupt 中断示例：模拟转账确认流程"""
import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing import TypedDict


class GraphState(TypedDict):
    """图的状态"""
    action_summary: str  # 待确认的操作描述
    user_input: str      # 用户在终端里输入的确认文本


def show_transfer(state: GraphState) -> dict:
    """展示一笔待确认的转账（模拟，不会真扣款）"""
    return {"action_summary": "向张三转账 ¥100（模拟，不会真扣款）"}


def wait_confirm(state: GraphState) -> dict:
    """
    中断节点：暂停图执行，等待用户在终端里输入确认。
    调用 interrupt() 后图会挂起，直到调用者用 Command(resume=...) 恢复。

    这个字典是给调用者看的提示信息，不是给图内部用的。图挂起后，调用者从返回值里取出来展示给用户：
    """
    text = interrupt({
        "hint": "终端里输入「确认」或备注后回车，图才会继续",
        "action_summary": state["action_summary"],
    })
    return {"user_input": str(text)}


# 构建图：START -> showTransfer -> waitConfirm -> END
graph = (
    StateGraph(GraphState)
    .add_node("show_transfer", show_transfer)
    .add_node("wait_confirm", wait_confirm)
    .add_edge(START, "show_transfer")
    .add_edge("show_transfer", "wait_confirm")
    .add_edge("wait_confirm", END)
    .compile(checkpointer=MemorySaver())
)


async def main() -> None:
    # 导出 Mermaid 图
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    config = {"configurable": {"thread_id": "interrupt-demo"}}

    # 第一次 invoke：会在 wait_confirm 处中断
    paused = await graph.ainvoke(
        {"action_summary": "", "user_input": ""},
        config,
    )
    print(f"paused: \n{paused}")
    # 取出中断信息展示给用户
    interrupt_info = paused.get("__interrupt__", [])
    if interrupt_info:
        print("\n待你确认：", interrupt_info[0].value)

    # 等待用户在终端输入
    '''
    Python 标准库内置，不需要 import。                                                                    
                                                                                                              
    line = input("> ").strip()
    #            ^^^                                                                                            
    #            提示符，打印在终端里
                                                                                                                
    - 程序暂停，等用户在终端输入并回车                                                                        
    - 返回用户输入的字符串（不含换行符）    
    - .strip() 去掉首尾空格 
    '''
    line = input("> ").strip()
    if not line:
        print("未输入，退出。")
        return

    # 用 Command(resume=...) 恢复图的执行
    '''
    Command 是 LangGraph 的控制指令对象，专门用来向图发送恢复信号：                                           
                                                                                                              
    Command(resume=line)                                                                                        
    #       ^^^^^^                                                                                              
    #       把用户输入的值传回给 interrupt() 的返回值                                                           
                                                                                                                
    ainvoke 第二次调用时，传的不是普通状态字典，而是 Command 对象，LangGraph                                  
    识别后知道这是"恢复挂起的图"，而不是"新启动一次图"。
    '''
    done = await graph.ainvoke(Command(resume=line), config)
    print("结果：", done)


if __name__ == "__main__":
    asyncio.run(main())
