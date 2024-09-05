from typing import Literal

import streamlit as st


def header(
    page_width: int, header_text: str, logo_path: str, logo_width: int = 230
) -> None:
    """Creates the app-wide header shown at login and above the core-page logic

    Args:
        page_width (int): Page width proportion. Takes value 1-100
        header_text (str): Text Displayed at the top of app.
        logo_path (str): file path to the logo. Logo should be housed in the assets folder
        logo_width (int, optional): px width of the logo file. Defaults to 250.
    """
    # Set max width param, hacky way but supported by streamlit technically:
    # https://discuss.streamlit.io/t/can-not-set-page-width-in-streamlit-1-5-0/21522/5
    css = (
        "<style>section.main > div {max-width:"
        + f"{page_width}"
        + "rem}</style>"
    )
    st.html(css)

    # hide red-yellow ribbon header
    hide_decoration_bar_style = """
        <style>
            header {visibility: hidden;}
        </style>
    """
    st.html(hide_decoration_bar_style)

    col_header, col_logo = st.columns([4, 1])
    with col_header:
        st.title(f"{header_text}")
    with col_logo:
        st.write(" ")
        st.image(logo_path, width=logo_width)

    st.markdown(" ## ")


def add_vertical_space(num_lines: int = 1) -> None:
    """Adds vertical space between two elements.

    Args:
        num_lines (int, optional): The number of h6 lines of spacing to add. Defaults to 1.

    Usage Example:
        ```python
        st.header("This is my header")
        add_vertical_space(num_lines=10)
        st.write("This text is far below the header")
        ```
    """
    for i in range(0, num_lines + 1):
        st.markdown(" ###### ")


def center_header_type(header: str, h: Literal[1, 2, 3, 4, 5, 6] = 1) -> None:
    """Creates a center aligned header of specified h1 through h6.
    Similar to st.markdown('##### {header}'), just aligned center within the container

    Args:
        header (str): Text to be displayed in the header
        h (Literal[1, 2, 3, 4, 5, 6]): HTML header level to be used. Must be one of [1, 2, 3, 4, 5, 6].

    Usage Example:
        ```python
        center_header_type(header="This h4 header is centered", h = 4)
        st.markdown("#### This h4 header is left aligned")
        ```
    """
    if h not in [1, 2, 3, 4, 5, 6]:
        raise ValueError("h must be one of [1, 2, 3, 4, 5, 6]")
    st.html(f"<h{h} style='text-align: center;'>{header}</h{h}>")


def center_header(header: str) -> None:
    """Creates a center aligned subheader.
    Similar to st.subheader, just aligned center on the page

    Args:
        header (str): Text to be displayed in the header

     Usage Example:
        ```python
        center_header("This header is centered")
        st.header("This header is left algined")
        ```
    """
    st.html(f"<h1 style='text-align: center;'>{header}</h1>")


def center_subheader(subheader: str) -> None:
    """Creates a center aligned subheader.
    Similar to st.subheader, just aligned center on the page

    Args:
        subheader (str): Text to be displayed in the subheader

    Usage Example:
        ```python
        center_subheader("This subheader is centered")
        st.subheader("This subheader is left algined")
        ```
    """
    st.html(f"<h3 style='text-align: center;'>{subheader}</h3>")


def center_text(text: str) -> None:
    """Creates a center aligned text block.
    similar to st.write, just centered.

    Args:
        text (str): text to be displayed

    Usage Example:
        ```python
        center_text("This text is centered")
        st.write("This text is left algined")
        ```
    """
    st.html(f"<div style='text-align: center;'>{text}</div>")


def center_button() -> None:
    """
    Center aligns button objects.
    Currently, this would effects ALL st.button on a page/container

    Usage Example:
        ```python
        center_button()
        st.button("Click Me!")
        ```
    """
    # Apply CSS to center align the button
    st.html(
        f"""
        <style>
        .stButton {{
            text-align: center;
        }}
        </style>
        """
    )


def style_metric_cards_dark_theme() -> None:
    """Styles the metric cards against a dark theme app. Use in conjuction with the dark theme in .streamlit/config.toml
        [theme]
        base="dark"

    Usage Example:
        ```python
        style_metric_cards_dark_theme()
        st.metric(label = "Dark Metric Card", value = 123.45)
        ```
    """
    from streamlit_extras.metric_cards import style_metric_cards

    style_metric_cards(
        border_left_color="#006EC7",
        background_color="#FFFFF",
        border_color="#fafafa33",
        box_shadow=True,
    )


def style_metric_cards_light_theme() -> None:
    """Styles the metric cards against a light theme app. Use in conjuction with the light theme in .streamlit/config.toml
        [theme]
        base="light"

    Usage Example:
        ```python
        style_metric_cards_light_theme()
        st.metric(label = "Light Metric Card", value = 123.45)
        ```
    """
    from streamlit_extras.metric_cards import style_metric_cards

    style_metric_cards(border_left_color="#006EC7")


def DMSansFont() -> None:
    """Function that sets app wide font to DM sans. Known to work on text, headers, plotly, forms, etc.
    Currently does not apply to text within st.dataframe

    This function is already called within the auth handling logic, so it is unlikely you will need to call it on your script.

    Usage Example:
        ```python
        DMSansFont()
        st.write("This text is in DM Sans!")
        ```
    """
    font_style = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

        html, body * {
            font-family: 'DM Sans', sans-serif !important;
        }
        """
    st.html(font_style)


def drop_shadow_container(css_class: str) -> None:
    """Apply drop shadow styling to containers. Best used in combination with border=True to make cool looking cards and dashboard effect.

    Args:
        css_class (str): css class of the st.emotion-cache

    Usage Example:
        ```python
        drop_shadow_container(css_class="st-emotion-cache-r421ms e1f1d6gn0")
        with st.container(border = True):
            st.subheader("this is my example card")
        ```
    """
    # remove space if copied directly.
    css_class = css_class.replace(" ", ".")
    st.html(
        f"""
        <style> 
            div.{css_class} {{
            box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
            }}
        </style>
    """
    )


def right_align_metrics() -> None:
    """Right aligns an st.metric
    Currently untested on style_metrics_cards

    Usage Example:
        ```python
        right_align_metrics()
        st.metric(label = "Example Metric", value = 123.45)
        ```
    """
    css = """
    <style>
        /* Align the main column to the right */
        div[data-testid="column"]:nth-of-type(2) {
            text-align: right;
        }

        /* Align the metric container to the right */
        div[data-testid="stMetric"] {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }

        /* Align the metric label to the right */
        div[data-testid="stMetricLabel"] p {
            text-align: right;
        }

        /* Align the metric value to the right */
        div[data-testid="stMetricValue"] {
            text-align: right;
        }

        /* Align the delta container to the right */
        div[data-testid="stMetricDelta"] {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            text-align: right;
        }

        /* Align the delta icon and value to the right */
        div[data-testid="stMetricDelta"] > svg,
        div[data-testid="stMetricDelta"] > div {
            text-align: right;
        }
    </style>
    """
    st.html(css)