Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        autoProcessQueue: false
    });

    dz.on("addedfile", function () {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function (file) {
        var url = "/classify_image"; // 🔥 FINAL FIX

        $.post(url, {
            image_data: file.dataURL
        }, function (data) {

            if (!data || data.length === 0) {
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;

            for (let i = 0; i < data.length; i++) {
                let maxScore = Math.max(...data[i].class_probability);
                if (maxScore > bestScore) {
                    match = data[i];
                    bestScore = maxScore;
                }
            }

            if (match) {
                $("#error").hide();
                $("#resultHolder").html($(`[data-player="${match.class}"]`).html());

                let dict = match.class_dictionary;
                for (let person in dict) {
                    let index = dict[person];
                    $("#score_" + person).html(match.class_probability[index]);
                }
            }
        });
    });

    $("#submitBtn").on('click', function () {
        dz.processQueue();
    });
}

$(document).ready(function () {
    $("#error").hide();
    init();
});
