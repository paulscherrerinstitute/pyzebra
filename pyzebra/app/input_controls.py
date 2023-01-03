import base64
import io
import os

from bokeh.io import curdoc
from bokeh.models import Button, FileInput, MultiSelect, Spinner

import pyzebra


class InputControls:
    def __init__(self, dataset, dlfiles, on_file_open=lambda: None, on_monitor_change=lambda: None):
        doc = curdoc()

        def filelist_select_update_for_proposal():
            proposal_path = proposal_textinput.name
            if proposal_path:
                file_list = []
                for file in os.listdir(proposal_path):
                    if file.endswith((".ccl", ".dat")):
                        file_list.append((os.path.join(proposal_path, file), file))
                filelist_select.options = file_list
                open_button.disabled = False
                append_button.disabled = False
            else:
                filelist_select.options = []
                open_button.disabled = True
                append_button.disabled = True

        doc.add_periodic_callback(filelist_select_update_for_proposal, 5000)

        def proposal_textinput_callback(_attr, _old, _new):
            filelist_select_update_for_proposal()

        proposal_textinput = doc.proposal_textinput
        proposal_textinput.on_change("name", proposal_textinput_callback)

        filelist_select = MultiSelect(title="Available .ccl/.dat files:", width=210, height=250)
        self.filelist_select = filelist_select

        def open_button_callback():
            new_data = []
            for f_path in self.filelist_select.value:
                with open(f_path) as file:
                    f_name = os.path.basename(f_path)
                    base, ext = os.path.splitext(f_name)
                    try:
                        file_data = pyzebra.parse_1D(file, ext)
                    except:
                        print(f"Error loading {f_name}")
                        continue

                pyzebra.normalize_dataset(file_data, monitor_spinner.value)

                if not new_data:  # first file
                    new_data = file_data
                    pyzebra.merge_duplicates(new_data)
                    dlfiles.set_names([base] * dlfiles.n_files)
                else:
                    pyzebra.merge_datasets(new_data, file_data)

            if new_data:
                dataset.clear()
                dataset.extend(new_data)
                on_file_open()
                append_upload_button.disabled = False

        open_button = Button(label="Open New", width=100, disabled=True)
        open_button.on_click(open_button_callback)
        self.open_button = open_button

        def append_button_callback():
            file_data = []
            for f_path in self.filelist_select.value:
                with open(f_path) as file:
                    f_name = os.path.basename(f_path)
                    _, ext = os.path.splitext(f_name)
                    try:
                        file_data = pyzebra.parse_1D(file, ext)
                    except:
                        print(f"Error loading {f_name}")
                        continue

                pyzebra.normalize_dataset(file_data, monitor_spinner.value)
                pyzebra.merge_datasets(dataset, file_data)

            if file_data:
                on_file_open()

        append_button = Button(label="Append", width=100, disabled=True)
        append_button.on_click(append_button_callback)
        self.append_button = append_button

        def upload_button_callback(_attr, _old, _new):
            new_data = []
            for f_str, f_name in zip(upload_button.value, upload_button.filename):
                with io.StringIO(base64.b64decode(f_str).decode()) as file:
                    base, ext = os.path.splitext(f_name)
                    try:
                        file_data = pyzebra.parse_1D(file, ext)
                    except:
                        print(f"Error loading {f_name}")
                        continue

                pyzebra.normalize_dataset(file_data, monitor_spinner.value)

                if not new_data:  # first file
                    new_data = file_data
                    pyzebra.merge_duplicates(new_data)
                    dlfiles.set_names([base] * dlfiles.n_files)
                else:
                    pyzebra.merge_datasets(new_data, file_data)

            if new_data:
                dataset.clear()
                dataset.extend(new_data)
                on_file_open()
                append_upload_button.disabled = False

        upload_button = FileInput(accept=".ccl,.dat", multiple=True, width=200)
        # for on_change("value", ...) or on_change("filename", ...),
        # see https://github.com/bokeh/bokeh/issues/11461
        upload_button.on_change("filename", upload_button_callback)
        self.upload_button = upload_button

        def append_upload_button_callback(_attr, _old, _new):
            file_data = []
            for f_str, f_name in zip(append_upload_button.value, append_upload_button.filename):
                with io.StringIO(base64.b64decode(f_str).decode()) as file:
                    _, ext = os.path.splitext(f_name)
                    try:
                        file_data = pyzebra.parse_1D(file, ext)
                    except:
                        print(f"Error loading {f_name}")
                        continue

                pyzebra.normalize_dataset(file_data, monitor_spinner.value)
                pyzebra.merge_datasets(dataset, file_data)

            if file_data:
                on_file_open()

        append_upload_button = FileInput(
            accept=".ccl,.dat", multiple=True, width=200, disabled=True
        )
        # for on_change("value", ...) or on_change("filename", ...),
        # see https://github.com/bokeh/bokeh/issues/11461
        append_upload_button.on_change("filename", append_upload_button_callback)
        self.append_upload_button = append_upload_button

        def monitor_spinner_callback(_attr, _old, new):
            if dataset:
                pyzebra.normalize_dataset(dataset, new)
                on_monitor_change()

        monitor_spinner = Spinner(title="Monitor:", mode="int", value=100_000, low=1, width=145)
        monitor_spinner.on_change("value", monitor_spinner_callback)
        self.monitor_spinner = monitor_spinner
