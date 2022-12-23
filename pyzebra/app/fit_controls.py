import types

from bokeh.models import (
    Button,
    CellEditor,
    CheckboxEditor,
    CheckboxGroup,
    ColumnDataSource,
    DataTable,
    Dropdown,
    MultiSelect,
    NumberEditor,
    RadioGroup,
    Spinner,
    TableColumn,
    TextAreaInput,
)

import pyzebra


def _params_factory(function):
    if function == "linear":
        param_names = ["slope", "intercept"]
    elif function == "gaussian":
        param_names = ["amplitude", "center", "sigma"]
    elif function == "voigt":
        param_names = ["amplitude", "center", "sigma", "gamma"]
    elif function == "pvoigt":
        param_names = ["amplitude", "center", "sigma", "fraction"]
    elif function == "pseudovoigt1":
        param_names = ["amplitude", "center", "g_sigma", "l_sigma", "fraction"]
    else:
        raise ValueError("Unknown fit function")

    n = len(param_names)
    params = dict(
        param=param_names, value=[None] * n, vary=[True] * n, min=[None] * n, max=[None] * n
    )

    if function == "linear":
        params["value"] = [0, 1]
        params["vary"] = [False, True]
        params["min"] = [None, 0]

    elif function == "gaussian":
        params["min"] = [0, None, None]

    return params


class FitControls:
    def __init__(self):
        self.params = {}

        def add_function_button_callback(click):
            # bokeh requires (str, str) for MultiSelect options
            new_tag = f"{click.item}-{function_select.tags[0]}"
            function_select.options.append((new_tag, click.item))
            self.params[new_tag] = _params_factory(click.item)
            function_select.tags[0] += 1

        add_function_button = Dropdown(
            label="Add fit function",
            menu=[
                ("Linear", "linear"),
                ("Gaussian", "gaussian"),
                ("Voigt", "voigt"),
                ("Pseudo Voigt", "pvoigt"),
                # ("Pseudo Voigt1", "pseudovoigt1"),
            ],
            width=145,
        )
        add_function_button.on_click(add_function_button_callback)
        self.add_function_button = add_function_button

        def function_list_callback(_attr, old, new):
            # Avoid selection of multiple indicies (via Shift+Click or Ctrl+Click)
            if len(new) > 1:
                # drop selection to the previous one
                function_select.value = old
                return

            if len(old) > 1:
                # skip unnecessary update caused by selection drop
                return

            if new:
                params_table_source.data.update(self.params[new[0]])
            else:
                params_table_source.data.update(dict(param=[], value=[], vary=[], min=[], max=[]))

        function_select = MultiSelect(options=[], height=120, width=145)
        function_select.tags = [0]
        function_select.on_change("value", function_list_callback)
        self.function_select = function_select

        def remove_function_button_callback():
            if function_select.value:
                sel_tag = function_select.value[0]
                del self.params[sel_tag]
                for elem in function_select.options:
                    if elem[0] == sel_tag:
                        function_select.options.remove(elem)
                        break

                function_select.value = []

        remove_function_button = Button(label="Remove fit function", width=145)
        remove_function_button.on_click(remove_function_button_callback)
        self.remove_function_button = remove_function_button

        params_table_source = ColumnDataSource(dict(param=[], value=[], vary=[], min=[], max=[]))
        self.params_table = DataTable(
            source=params_table_source,
            columns=[
                TableColumn(field="param", title="Parameter", editor=CellEditor()),
                TableColumn(field="value", title="Value", editor=NumberEditor()),
                TableColumn(field="vary", title="Vary", editor=CheckboxEditor()),
                TableColumn(field="min", title="Min", editor=NumberEditor()),
                TableColumn(field="max", title="Max", editor=NumberEditor()),
            ],
            height=200,
            width=350,
            index_position=None,
            editable=True,
            auto_edit=True,
        )

        # start with `background` and `gauss` fit functions added
        add_function_button_callback(types.SimpleNamespace(item="linear"))
        add_function_button_callback(types.SimpleNamespace(item="gaussian"))
        function_select.value = ["gaussian-1"]  # put selection on gauss

        self.from_spinner = Spinner(title="Fit from:", width=145)
        self.to_spinner = Spinner(title="to:", width=145)

        self.area_method_radiogroup = RadioGroup(labels=["Function", "Area"], active=0, width=145)

        self.lorentz_checkbox = CheckboxGroup(
            labels=["Lorentz Correction"], width=145, margin=(13, 5, 5, 5)
        )

        self.result_textarea = TextAreaInput(title="Fit results:", width=750, height=200)

    def _process_scan(self, scan):
        pyzebra.fit_scan(
            scan, self.params, fit_from=self.from_spinner.value, fit_to=self.to_spinner.value
        )
        pyzebra.get_area(
            scan,
            area_method=pyzebra.AREA_METHODS[self.area_method_radiogroup.active],
            lorentz=self.lorentz_checkbox.active,
        )

    def fit_scan(self, scan):
        self._process_scan(scan)

    def fit_dataset(self, dataset):
        for scan in dataset:
            if scan["export"]:
                self._process_scan(scan)

    def update_result_textarea(self, scan):
        fit = scan.get("fit")
        if fit is None:
            self.result_textarea.value = ""
        else:
            self.result_textarea.value = fit.fit_report()
