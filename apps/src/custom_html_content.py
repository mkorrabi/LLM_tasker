def get_header_html(title: str) -> str:
    return f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
            <img src="file/apps/assets/img/airview.png" style="max-width:10vw;">
            <h2 style="text-align:center;">{title}</h2>
            <img src="file/apps/assets/img/AI.png" style="max-width:10vw;">
        </div>
     """


def get_footer_html() -> str:
    return """
        <div id="g2s-footer" style="text-align: center; margin-top: 30px;">
            <p style="opacity: 0.5;">Airview</p>
        </div>
     """


def get_blank_space() -> str:
    return """
        <div style='margin: 5px 0;'></div>
     """
