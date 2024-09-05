from page_logic.chatbot_page import chatbot_page_logic
from polydelta_streamlit_helpers.run_page import (
    run_page,
)

run_page(page_logic_function=chatbot_page_logic)