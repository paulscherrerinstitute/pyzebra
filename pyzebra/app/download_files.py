from bokeh.models import Button, ColumnDataSource, CustomJS

js_code = """
let j = 0;
for (let i = 0; i < source.data['name'].length; i++) {
    if (source.data['content'][i] === "") continue;

    setTimeout(function() {
        const blob = new Blob([source.data['content'][i]], {type: 'text/plain'})
        const link = document.createElement('a');
        document.body.appendChild(link);
        const url = window.URL.createObjectURL(blob);
        link.href = url;
        link.download = source.data['name'][i] + source.data['ext'][i];
        link.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(link);
    }, 100 * j)

    j++;
}
"""


class DownloadFiles:
    def __init__(self, n_files):
        source = ColumnDataSource(
            data=dict(content=[""] * n_files, name=[""] * n_files, ext=[""] * n_files)
        )
        self._source = source

        label = "Download File" if n_files == 1 else "Download Files"
        button = Button(label=label, button_type="success", width=200)
        button.js_on_click(CustomJS(args={"source": source}, code=js_code))
        self.button = button

    def set_contents(self, contents):
        self._source.data.update(content=contents)

    def set_names(self, names):
        self._source.data.update(name=names)

    def set_extensions(self, extensions):
        self._source.data.update(ext=extensions)
