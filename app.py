from flask import Flask, render_template, request, redirect, flash

from backend import process_data

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
            flash('Invalid file type')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                solves_details, overall_pbs = process_data(file)
                return render_template("data.html", solves_details=solves_details, overall_pbs=overall_pbs)
            except NotImplementedError:
                flash('Unable to process file')
                return redirect(request.url)
    return render_template("index.html")


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=True)
