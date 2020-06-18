from bokeh.layouts import column
from bokeh.models import Panel, TextInput, Button, RadioButtonGroup

import pyzebra


def create():
    fileinput = TextInput()

    mode_radio_button_group = RadioButtonGroup(
        labels=["Adaptive Peak Detection", "Adaptive Dynamic Mask Integration"], active=0
    )

    def process_button_callback():
        pyzebra.anatric(fileinput.value)

    process_button = Button(label="Process")
    process_button.on_click(process_button_callback)

    tab_layout = column(fileinput, mode_radio_button_group, process_button)

    return Panel(child=tab_layout, title="Anatric")
