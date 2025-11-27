import json, base64, logging, sqlite3, requests
import datetime as dt
from typing import Literal, Annotated
from typing_extensions import TypedDict
from pathlib import Path
from prompts import prompt_template, summary_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages.utils import trim_messages
from langchain_core.messages import (AnyMessage, HumanMessage, AIMessage, BaseMessage,
                                     SystemMessage, ToolMessage, RemoveMessage, AIMessageChunk)
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
#from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch, TavilyExtract

database_path = Path("db")
database_path.mkdir(exist_ok=True)
database_file = database_path / "history.db"

logger = logging.getLogger("__main__")

def fetch_encode_image(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('content-type', 'image/jpeg')
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        image_str =  f"data:{content_type};base64,{image_base64}"
        return image_str
    except Exception as e:
        logger.exception(f"Error fetching image for {url}")
        return None


class MessageType(TypedDict):
    role: Literal["assistant", "user", "tool"]
    mtype: Literal["text", "file", "search", "image", "error"]
    name: str
    content: str | dict  
    data: str | None
    
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    
class ChatAgent:

    def __init__(self, model_config: dict[str, any]):

        self.temperature = model_config['temperature']
        self.history_length_threshold = model_config['summarization_threshold']

        # init tools
        search_tool = TavilySearch(max_results=4,
                                   include_answer=True,
                                   include_images=True,
                                   include_image_descriptions=False)
        extract_tool = TavilyExtract(extract_depth="basic", include_images=False)
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        # init models
        llm = init_chat_model(model_config['llm'],
                              model_provider='google-genai',
                              temperature=model_config['temperature'])
        
        self.summary_model = init_chat_model(model_config['summary'],
                                             model_provider='google-genai',
                                             temperature=0.0)
        
        self.image_model = ChatGoogleGenerativeAI(model=model_config['image'])

        @tool("generate_image")
        def generate_image(image_descr: str) -> list[dict[str,any]]:
            """Generate an image for a given textual description"""
        
            try:
                response = self.image_model.invoke(image_descr,
                    generation_config=dict(response_modalities=["TEXT", "IMAGE"]))
            except Exception as e:
                logger.exception(f"Unable to generate image")
                return []
        
            logger.debug(f"generate_image: response type = {type(response.content)}")

            image_text = response.content[0]
            image_base64 = response.content[1].get("image_url").get("url")

            return [{"type": "text", "content": image_text},
                    {"type": "image_url", "image_url": {"url": image_base64 }}]

        self.tools = [search_tool, extract_tool, wiki_tool, generate_image]
        self.llm = llm.bind_tools(self.tools)
        self.app = self._create_workflow()

        self.tool_handlers = {
            # handle by name of tool
            "tavily_search": self._handle_search_tool,
            "tavily_extract": self._handle_text_tool,
            "wikipedia": self._handle_text_tool,
            "generate_image": self._handle_image_tool}

    def _create_workflow(self):
        """Create langgraph graph """
        
        def call_model(state: State) -> State:
            """ Main interaction with llm """

            summary = state.get("summary", None)
            messages = state.get("messages", None)
            if not messages:
                raise ValueError("No messages list")
            
            now = dt.datetime.now()
            current_date = now.strftime("%B %d %Y")
            current_time = now.strftime("%H:%M:%S")

            try:
                prompt = prompt_template.invoke({"current_date": current_date,
                                                 "current_time": current_time,
                                                 "summary": summary,
                                                 "messages": messages})
                response = self.llm.invoke(prompt.to_messages())
            except Exception as e:
                logger.exception(f"Error on llm invoke")
                return {}

            return {"messages": response }

        def summarize_messages(state: State) -> State:
            """Summarize the list of messages into a one paragraph string"""

            summary = state.get("summary", None)
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("No messages list")
           
            try:
                summary_prompt = summary_template.invoke({"messages": messages,
                                                      "existing_summary": summary})
                logger.debug(f"Summarize a list of {len(messages)} messages")
                response = self.summary_model.invoke(summary_prompt)
                logger.debug(f"summary: {response.content}")
            except Exception as e:
                logger.exception("Unable to query summary model")
                return {}

            logger.debug(f"trim {len(messages)} messages to 4")
            trimmed_messages = trim_messages(messages,
                                             strategy="last",
                                             token_counter=len,
                                             start_on="human",
                                             max_tokens=4)

            logger.debug(f"trimmed messages to {len(trimmed_messages)} messages")
            new_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + trimmed_messages

            return {"summary": response.content, "messages": new_messages}
        
        def decide_post_action(state: State) -> str:
            """decide next step: tools, summarize or END"""

            last_message = state['messages'][-1]
            if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
                return "tools"
            
            elif len(state['messages']) > self.history_length_threshold:
                return "summarize_messages"
            return END
            
        self.tool_node = ToolNode(self.tools, handle_tool_errors=True)

        try: 
            # self.memory = InMemorySaver()
            self.conn = sqlite3.connect(database_file, check_same_thread=False)
            self.memory = SqliteSaver(self.conn)
        except Exception as e:
            logger.exception(f"Unable to open database connection for {database_file}")
            return None

        workflow = StateGraph(State)
        workflow.add_node("call_model", call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("summarize_messages", summarize_messages)
        workflow.add_edge(START, "call_model")
        workflow.add_conditional_edges("call_model",
                                       decide_post_action,
                                       {"tools": "tools",
                                        "summarize_messages": "summarize_messages",
                                        END:END})
        workflow.add_edge("tools", "call_model")
        workflow.add_edge("summarize_messages", END)
        return workflow.compile(checkpointer=self.memory)

    def get_chat_length(self, config: RunnableConfig) -> int:
        """Return the length of the chat for user config"""
        try:
            if snapshot := self.app.get_state(config):
                messages = snapshot.values.get('messages', [])
                return len(messages)
        except ValueError:
            # This can happen if no checkpointer is configured.
            logger.warning("No checkpointer found. Cannot get chat length.")
            return -1

    def update_llm(self, model_config: dict) -> None:
        """Update LLM with now changes to model_config"""

        if model_config['temperature'] != self.temperature:
            self.llm = init_chat_model(model_config['llm'],
                                       model_provider='google-genai',
                                       temperature=model_config['temperature'])

            self.temperature = model_config['temperature']

    def _handle_search_tool(self, msg: ToolMessage) -> MessageType:
        """handle ToolMessage response from tavily_search"""

        try:
            content = json.loads(msg.content)
            image_urls = content.get("images", [])
            logger.debug(f"search tool found {len(image_urls)} images")
            retrieved_images = []
            for url in image_urls:
                image_base64  =  fetch_encode_image(url)
                if not image_base64:
                    logging.debug(f"cannot fetch {url}")
                    continue
                logging.debug(f"image at {url}")
                retrieved_images.append((url, image_base64))
            content['images']  = retrieved_images
        except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
            logger.exception(f"Unable to parse json content: {msg.content}")
            return {"role": "tool", "mtype": "error", "name": msg.name, "content": "Error: unable to parse json content"}

        return {"role": "tool", "mtype": "search", "name": msg.name, "content": content}
        
    def _handle_text_tool(self, msg: ToolMessage) -> MessageType:
        """Generic handler for simple text-based tools."""

        return {"role": "tool", "mtype": "text", "name": msg.name, "content": msg.content}
    
    def _handle_image_tool(self, msg: ToolMessage) -> MessageType:
        """Handler for the image generation tool."""

        if not isinstance(msg.content, list):
            logger.error(f"generate_image tool did not return a list: {msg.content}")
            # Optionally, you could yield an error message to the UI here
            return {"role": "tool", "mtype": "error", "name": msg.name,
                    "content": f"Error: ToolMessage content not a list: {msg.content}"}
    
        try:
            text_content = msg.content[0]['content']
            image_data = msg.content[1]['image_url']['url']
        except (IndexError, KeyError, ValueError) as e:
            logger.exception("Invalid structure content field")
            error_msg = f"Error: Invalid structure in image tool response: {msg.content}"
            return {"role": "tool", "mtype": "error", "name": msg.name, "content": {error_msg}}

        return  {"role": "tool", "mtype": "image", "name": msg.name,"content": text_content,"data": image_data}

    def stream_response(self, user_message, config: RunnableConfig):
        """stream response from graph """
        
        for chunk in self.app.stream(user_message, config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "call_model":
                    if update:
                        msg = update.get("messages", [])
                        logger.debug(f"stream {type(msg)} from {node_name}")
                        if msg and isinstance(msg, AIMessage):
                            logger.debug(f"stream llm msg: {msg.content[:30]}")
                            yield {"role": "assistant", "mtype": "text", "name": "llm", "content": msg.content}
                        else:
                            logging.error(f"Uknown message type {type(ai_msg)}")
                    
                elif node_name == "tools":
                    if update:
                        sub = update.get("messages", [])
                        logger.debug(f"stream {type(sub)} from {node_name}")
                        if isinstance(sub, ToolMessage):
                            handler = self.tool_handlers.get(sub.name)
                            if handler:
                                yield handler(sub)
                            else:
                                logger.warning(f"No handler found for tool: {sub.name}")
                        elif isinstance(sub, list):
                            for msg in sub:
                                handler = self.tool_handlers.get(msg.name)
                                if handler:
                                    yield handler(msg)
                                else:
                                    logger.warning(f"No handler found for tool: {msg.name}")
                        else:
                            logging.error(f"Unknown message type {type(sub)}")
                else:
                    logger.error(f"update from unknown node {node_name}")

