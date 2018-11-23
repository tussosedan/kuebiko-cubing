from flask import Flask, render_template, request, redirect, flash, Markup

from backend import process_data

import json

app = Flask(__name__)
app.secret_key = "super secret key !@#"
ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Please select a file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Please select a file')
            return redirect(request.url)
        if file and not allowed_file(file.filename):
            flash(Markup('Looks like this file type is not supported yet. '
                         'Please open an issue on the '
                         '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                         ' and upload the file there.'))
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                solves_details, overall_pbs, timer_type, datalen = process_data(file)
                return render_template("data.html", solves_details=solves_details, overall_pbs=overall_pbs,
                                       timer_type=timer_type, datalen=datalen)
            except NotImplementedError:
                flash(Markup('Looks like this file type is not supported yet. '
                             'Please open an issue on the '
                             '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                             ' and upload the file there.'))
                return redirect(request.url)
            except (json.decoder.JSONDecodeError, ValueError, KeyError):
                flash(Markup('Something went wrong while reading the file. '
                             'Please open an issue on the '
                             '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                             ' and upload the file there.'))
                return redirect(request.url)
    return render_template("index.html")


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=True)
    # app.run()
