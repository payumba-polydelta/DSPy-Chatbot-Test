from typing import Callable


def run_page(page_logic_function: Callable[[], None]) -> None:
    """Authenticate users and execute the provided page logic.

    This function handles user authentication, sets up page configuration,
    and applies base styling before executing the provided page logic.

    This function should be used on any file which leads users to a page in streamlit App, entry point or multipage

    YOU SHOULD NOT EDIT THIS FUNCTION IF CLONING FROM THE TEMPLATE

    Args:
        page_logic_function (Callable): function of the user's page as defined in page_logic directory

    Usage Example

        ```python
        from polydelta_streamlit_helpers.authenticate_and_run_page import authenticate_and_run_page
        from page_logic.page import my_page

        authenticate_and_run_page(page_logic_function=my_page)
        ```
    """
    import streamlit as st

    from polydelta_streamlit_helpers.utils import load_yaml

    params = load_yaml("params.yaml")
    # Page config has to be set before any other streamlit commands, such as caching!
    st.set_page_config(
        page_title=params["page_config"]["page_title"],
        page_icon=params["page_config"]["page_icon"],
        initial_sidebar_state=params["page_config"]["initial_sidebar_state"],
    )

    from polydelta_streamlit_helpers.style import DMSansFont, header

    DMSansFont()

    header(
        page_width=params["header"]["page_width"],
        header_text=params["header"]["header_text"],
        logo_path=params["header"]["logo_path"],
        logo_width=params["header"]["logo_width"],
    )
    page_logic_function()  # run your page