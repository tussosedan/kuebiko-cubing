from flask import Flask, render_template, request, redirect, flash, Markup
from backend import process_data
import os
import traceback
import time
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'txt'}
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
            # noinspection PyBroadException
            try:
                chart_by = request.form.get('chart-by', 'chart-by-dates')
                timezone = request.form.get('tz', 'UTC')
                solves_details, overall_pbs, solves_by_dates, timer_type, datalen = process_data(file, chart_by,
                                                                                                 timezone)
                return render_template("data.html", solves_details=solves_details, overall_pbs=overall_pbs,
                                       solves_by_dates=solves_by_dates, timer_type=timer_type, datalen=datalen)
            except NotImplementedError:
                flash(Markup('Looks like this file type is not supported yet. '
                             'Please open an issue on the '
                             '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                             ' and upload the file there.'))
                return redirect(request.url)
            except Exception:
                flash(Markup('Something went wrong while reading the file. '
                             'Please open an issue on the '
                             '<a href="https://github.com/tussosedan/kuebiko-cubing/issues">github page</a>'
                             ' and upload the file there.'))
                timestr = time.strftime("%Y%m%d_%H%M%S_")
                filename = timestr + secure_filename(file.filename)
                file.seek(0)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                err_filename = filename + '_err'
                with open(os.path.join(app.config['UPLOAD_FOLDER'], err_filename), 'w') as err_file:
                    err_file.write(traceback.format_exc())
                return redirect(request.url)
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
