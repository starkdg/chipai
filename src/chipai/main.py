import sys, base64, os, json, time, requests, logging
from dotenv import load_dotenv
from json import JSONDecodeError
import datetime as dt
from pathlib import Path
import streamlit as st
from agent import ChatAgent, MessageType
from langchain_core.messages import HumanMessage

# load environment variables
load_dotenv()

LOGGING_PATH = "logs"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# silence loggers used by streamlit
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):  
    logging_path = Path(LOGGING_PATH)
    logging_path.mkdir(exist_ok=True)
    log_file = logging_path / "agent.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)-5s]: %(message)s [%(funcName)s:%(lineno)d]")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def init_session_state():
    """Init session state variables"""
    logger.info(f"Current working directory: {os.getcwd()}")

    if "model_config" not in st.session_state:

        try: 
            models_config = st.secrets.get('models')
            agent_config = st.secrets.get('agent')
        except FileNotFoundException as e:
            st.error("Your .streamlit/secrets.toml file appears to be missing.")
            st.stop()
        except KeyError as e:
            st.error("Your .streamlit/secrets.toml file is missing [models] and [agent] sections")
            st.stop()
            
        required_models = ['GEMINI_MODEL', 'SUMMARY_MODEL', 'IMAGE_MODEL']
        if not all(key in models_config for key in required_models):
            st.error("Your .streamlit/secrets.toml file is missing one or more required keys under the [models] section.")
            st.info(f"Please ensure all of the following are present: {required_models}")
            st.stop()

        st.session_state.model_config = {
            "llm": models_config.get('GEMINI_MODEL'),
            "summary": models_config.get('SUMMARY_MODEL'),
            "image":  models_config.get('IMAGE_MODEL'),
            'temperature': 0.30,
            "summarization_threshold": agent_config.get('SUMMARIZATION_THRESHOLD', 4),
        }
        st.session_state.temperature = st.session_state.model_config['temperature']
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    if "chat_history_length" not in st.session_state:
        st.session_state.chat_history_length = 0

    if "agent" not in st.session_state:
        st.session_state.agent = get_chat_agent(st.session_state.model_config)

def add_message(msg: MessageType) -> None:
    """add a message to list """

    if len(st.session_state["messages"]) >= 100:
        st.session_state["messages"].pop(0)

    st.session_state["messages"].append(msg)

def _handle_search_message(msg: MessageType, container: st.container) -> None:
    """print dict item from a tavily search result """

    with container.chat_message(msg['role']):
        content = msg.get('content', {})
        if not isinstance(content, dict):
            st.error(f"Search result error: {content}")
            return
        
        st.markdown(f"**Query:** {content.get('query')}")
        st.markdown(f"**Answer:** {content.get('answer')}")
        st.markdown(f"Found {len( content.get('results', []))} links:")
        for i, element in enumerate(content.get('results', [])):
            st.markdown(f"""
            **{i+1}. Score: {element.get('score')}**
            > {element.get('content')}
            *Source: {element.get('url')}*
            """)

        images = content.get('images', [])
        st.write(f"Found {len(images)} images")
        for url, image_data in images:
            if image_data:
                st.image(image_data, caption=url, width=300)

def _handle_extract_message(sub: MessageType, container: st.container) -> None:

    try:
        content = sub['content'].strip()
        result = json.loads(content.strip())
        for item in result['results']:
            container.chat_message(sub['role']).write(item['raw_content'])
    except (JSONDecodeError, UnicodeDecodeError, Exception) as e:
        logger.error(f"Invalid json: {content}")
        logger.exception(e)
        container.chat_message(sub['role']).markdown("Error decoding content")

# handler for messages with pure text
def _handle_text_message(sub: MessageType, container: st.container) -> None:

    if sub['name'] == "tavily_extract":
        _handle_extract_message(sub, container)
    else:
        container.chat_message(sub['role']).write(sub['content'])

def _handle_image_message(sub: MessageType, container: st.container) -> None:

    with container:
        st.chat_message(sub['role']).image(sub['data'], caption=sub['content'], width=600)

# key is the name field of the message, value is the function handler
message_handlers = {
    # msg handlers by mtype
    "text" : _handle_text_message,
    "file" : _handle_text_message,
    "image" : _handle_image_message,
    "search" : _handle_search_message,
    "error" : _handle_text_message,
}

def print_message_list(message_list: list[MessageType], container) -> None:
    """print message list to container"""

    for msg in message_list:
        handler = message_handlers.get(msg['mtype'])
        if handler:
            handler(msg, container)
        else:
            logger.error(f"Unknown msg in message list, name = {msg['name']} type = {msg['mtype']}")

def process_uploaded_files(prompt, container) -> list[dict]:
    """process the uploaded files """

    prompt_content = [{"type": "text", "text": prompt.text }]
    for uploaded_file in prompt.files:
        file_type = uploaded_file.type.split('/')
        file_bytes = uploaded_file.getvalue()
        file_data = base64.b64encode(file_bytes).decode('utf-8')
        base64_data = f"data:{uploaded_file.type};base64,{file_data}"
        if uploaded_file.type.startswith("image"):
            # image files (e.g. png, jpeg, webp)
            prompt_content.append({"type": file_type[0], "source_type": "base64", "mime_type": uploaded_file.type, "data": file_data})
            try:
                container.image(base64_data, caption=uploaded_file.name, width=450)
            except Exception as e:
                logger.exeception("Unable to display image")
                add_message({"role": "user", "mtype": "error", "name": "image", "content": f"Error:{e}"})
                continue
            add_message({"role": "user", "mtype": "image", "name": "upload", "content": uploaded_file.name, "data": base64_data})
        elif uploaded_file.type == "application/pdf" or uploaded_file.type.startswith("text"):
            # text file (pdf, py, org, txt)
            prompt_content.append({"type": "file", "source_type": "base64", "mime_type": uploaded_file.type, "data": file_data})
            container.write(f"uploaded file: {uploaded_file.name}")
            add_message({"role": "user", "mtype": "file", "name": "upload", "content": uploaded_file.name})
        else:
            error_str = f"Error: uploaded file {uploaded_file.name} of unrecognized type, {uploaded_file.type}"
            logging.info(error_str)
            add_message({"role": "user", "mtype": "error", "name": "upload", "content": error_str})

    return prompt_content

@st.cache_resource(ttl=3600, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)})
def get_chat_agent(model_config: dict[str, str]):
    """Create and cache an agent"""

    try: 
        agent = ChatAgent(model_config) 
        return agent
    except (ValueError, ImportError) as e:
        logger.exception(f"Unable to initialize ChatAgent")
        st.stop()
    
def update_agent():
    """update the agent when the model_config changes (e.g. the temperature slider)"""
    logger.info("updating agent with new temperature = %f", st.session_state.model_config['temperature'])
    st.session_state.agent = get_chat_agent(st.session_state.model_config)

  
def main():

    init_session_state()
    st.logo("assets/logo.png", size="large")

    st.session_state.temperature = st.sidebar.slider("Temperature", min_value=0.0,
                                                     max_value=1.0, step=0.05,
                                                     value=0.30)
    if st.session_state.temperature != st.session_state.model_config['temperature']:
        st.session_state.model_config['temperature'] = st.session_state.temperature
        update_agent()
    
    # login
    if not st.user.is_logged_in:
        st.info("Please log in to continue.")
        st.sidebar.button("Login with Google", on_click=st.login, args=["google"])
        st.stop()
    
    # logout
    if st.sidebar.button("Logout"):
        st.logout()
        st.rerun()

    config = {"configurable": {"thread_id": st.user['email']}}
    st.sidebar.markdown(f"Logged in as {st.user['email']}")
    st.sidebar.divider()

    st.session_state.chat_history_length = st.session_state.agent.get_chat_length(config)
    st.sidebar.write("Chat History Length: ", st.session_state.chat_history_length)
    st.sidebar.divider()

    chat_container = st.container()
    with chat_container:
        st.image("assets/chipai.png", width=400)
        
    print_message_list(st.session_state['messages'], chat_container)
    
    RPM = 15 # requests per minute, rpm
    RATE_LIMIT = 60//RPM # rate limit interval in seconds
    current_time = time.time()
    time_since_last_request = current_time - st.session_state.last_request_time

    if prompt := st.chat_input(placeholder="Say something", accept_file="multiple", file_type=None):

        chat_container.chat_message("user").write(prompt.text)
        add_message({"role":"user", "mtype":"text", "name": "user_message", "content": prompt.text})

        if len(prompt.files) > 0: # attach uploaded files to content
            prompt_content = process_uploaded_files(prompt, chat_container)
        else:
            prompt_content = prompt.text
            
        user_message = HumanMessage(content=prompt_content)

        if time_since_last_request < RATE_LIMIT:
            delay = RATE_LIMIT - time_since_last_request
            logger.warning("Rate limit exceeded. Please wait %d seconds", RATE_LIMIT - time_since_last_request)
            time.sleep(delay)

        st.session_state.last_request_time = current_time

        logger.debug("new user prompt submitted: %s", prompt.text[:50])
            
        # Manage streaming responses
        with st.spinner("..."):
            for sub in st.session_state.agent.stream_response({"messages": user_message}, config):
                logger.debug(f"stream response from {sub['name']} of type {sub['mtype']}")
                handler = message_handlers.get(sub['mtype'])
                if handler:
                    handler(sub, chat_container)
                else:
                    logger.warning(f"No handler for this message name={sub['name']} mtype={sub['mtype']}")
                add_message(sub)

        st.session_state.chat_history_length = st.session_state.agent.get_chat_length(config)
        st.rerun()
    
if __name__ == "__main__":
    main()
            
                


