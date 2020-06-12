from bokeh.layouts import column
from bokeh.models import Panel, TextInput, Button

import pyzebra


def create():
    fileinput = TextInput()
    process_button = Button(label="Process")

    def process_button_callback():
        pyzebra.anatric(fileinput.value)

    process_button.on_click(process_button_callback)
    
    tab_layout = column(fileinput, process_button)
    return Panel(child=tab_layout, title="Anatric")
