from flask import Flask, render_template, request, redirect, flash, Markup
from backend import process_data
import os
import traceback
import time
from werkzeug.utils import secure_filename
from io import BytesIO

ALLOWED_EXTENSIONS = {'txt', 'json', 'csv'}
UPLOAD_FOLDER = r'C:\uploads'

app = Flask(__name__)
app.secret_key = "super secret key !@#"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part or textinput
        if 'file' not in request.files and request.form.get('textinput', '') == '':
            flash('Please select a file or paste the data')
            return redirect(request.url)
        file = request.files.get('file', None)
        textdata = request.form.get('textinput', None)
        if textdata == '':
            textdata = None

        # if user does not select file, browser also
        # submit an empty part without filename
        if file and file.filename == '' and not textdata:
            flash('Please select a file or paste the data')
            return redirect(request.url)
        if file and not allowed_file(file.filename):
            flash(Markup('Looks like this timer''s  data is not supported yet. '
                         'Please open an issue on the '
                         '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                         ' and upload the file there.'))
            return redirect(request.url)
        if (file and allowed_file(file.filename)) or textdata:
            if file:
                file_to_send = file.stream
            else:
                file_to_send = BytesIO(textdata.encode())

            chart_by = request.form.get('chart-by', 'chart-by-dates')
            secondary_y_axis = request.form.get('secondary-y-axis', 'none')
            subx_threshold_mode = request.form.get('subx-threshold', 'auto')
            subx_override = request.form.get('subx-override', 'none')
            timezone = request.form.get('tz', 'UTC')

            if secondary_y_axis == 'none':
                secondary_y_axis = None

            # noinspection PyBroadException
            try:
                solves_details, overall_pbs, solves_by_dates, timer_type, datalen = \
                    process_data(file_to_send, chart_by, secondary_y_axis, subx_threshold_mode, subx_override, timezone)
                return render_template("data.html", solves_details=solves_details, overall_pbs=overall_pbs,
                                       solves_by_dates=solves_by_dates, timer_type=timer_type, datalen=datalen)
            except NotImplementedError:
                flash(Markup('Looks like this file type is not supported yet. '
                             'Please open an issue on the '
                             '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                             ' and upload the file there.'))
                return redirect(request.url)
            except Exception:
                if not app.debug:
                    flash(Markup('Something went wrong while reading the file. '
                                 'Please open an issue on the '
                                 '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                                 ' and upload the file there.'))
                    timestr = time.strftime("%Y%m%d_%H%M%S_")
                    if file:
                        filename = timestr + secure_filename(file.filename)
                    else:
                        filename = timestr + "textarea"
                    file_to_send.seek(0)
                    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as data_file:
                        data_file.write(file_to_send.read())

                    err_filename = filename + '_err'
                    with open(os.path.join(app.config['UPLOAD_FOLDER'], err_filename), 'w') as err_file:
                        err_file.write(traceback.format_exc())
                    return redirect(request.url)
                else:
                    raise
    return render_template("index.html")


# noinspection PyUnusedLocal
@app.errorhandler(413)
def request_entity_too_large(e):
    flash(Markup('The file is too large. '
                 'If the file is truly valid, please open an issue on the '
                 '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                 ' and upload the file there.'))
    return render_template("index.html"), 413


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=True)
    # app.run()
