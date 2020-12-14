import tempfile

from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Panel,
    Spinner,
    TextAreaInput,
    TextInput,
)


def create():
    selection_list = TextAreaInput(title="ROIs:", rows=7)
    lattice_const_textinput = TextInput(title="Lattice constants:")
    max_res_spinner = Spinner(title="max-res", value=2, step=0.01)
    seed_pool_size_spinner = Spinner(title="seed-pool-size", value=5, step=0.01)
    seed_len_tol_spinner = Spinner(title="seed-len-tol", value=0.02, step=0.01)
    seed_angle_tol_spinner = Spinner(title="seed-angle-tol", value=1, step=0.01)
    eval_hkl_tol_spinner = Spinner(title="eval-hkl-tol", value=0.15, step=0.01)

    def process_button_callback():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = temp_dir + "/temp.xml"

    process_button = Button(label="Process", button_type="primary")
    process_button.on_click(process_button_callback)

    output_textarea = TextAreaInput(title="Output UB matrix:", rows=7)

    tab_layout = row(
        column(
            selection_list,
            lattice_const_textinput,
            max_res_spinner,
            seed_pool_size_spinner,
            seed_len_tol_spinner,
            seed_angle_tol_spinner,
            eval_hkl_tol_spinner,
            process_button,
        ),
        output_textarea,
    )

    return Panel(child=tab_layout, title="spind")
