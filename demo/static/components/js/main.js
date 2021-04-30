/*
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
*/

// Refer to https://github.com/quolc/neural-collage/blob/master/static/demo_feature_blending/js/main.js

max_colors = 3;
colors = ["#FE2712", "#66B032", "#FEFE33"];
original_image = null;
palette = [];
palette_selected_index = null;

ui_uninitialized = true;
p5_input_original = null;
p5_input_reference = null;
p5_output = null;
sync_flag = true;
id = null;

function ReferenceNameSpace() {
    return function (s) {
        s.setup = function () {
            s.pixelDensity(1);
            s.createCanvas(canvas_size, canvas_size);

            s.mask = [];
            for (var i = 0; i < max_colors; i++) {
                s.mask.push(s.createGraphics(canvas_size, canvas_size));
            }

            s.body = null;
            s.cursor(s.HAND);

        }

        s.draw = function () {
            s.background(255);
            s.noTint();
            if (s.body != null) {
                s.image(s.body, 0, 0, s.width, s.height);
            }
            s.tint(255, 127);

            if (palette_selected_index != null)
                s.image(s.mask[palette_selected_index], 0, 0);
        }

        s.mouseDragged = function () {
            if (ui_uninitialized) return;

            var c = $('.palette-item.selected').data('class');
            if (c != -1) {
                var col = s.color(colors[palette.indexOf(c)]);
                s.mask[palette_selected_index].noStroke();
                s.mask[palette_selected_index].fill(col);
                s.mask[palette_selected_index].ellipse(s.mouseX, s.mouseY, 20, 20);

            } else { // eraser
                if (sync_flag == true) {
                    var col = s.color(0, 0);
                    erase_size = 20;
                    s.mask[palette_selected_index].loadPixels();
                    for (var x = Math.max(0, Math.floor(s.mouseX) - erase_size); x < Math.min(canvas_size, Math.floor(s.mouseX) + erase_size); x++) {
                        for (var y = Math.max(0, Math.floor(s.mouseY) - erase_size); y < Math.min(canvas_size, Math.floor(s.mouseY) + erase_size); y++) {
                            if (s.dist(s.mouseX, s.mouseY, x, y) < erase_size) {
                                s.mask[palette_selected_index].set(x, y, col);
                            }
                        }
                    }
                    s.mask[palette_selected_index].updatePixels();

                    // p5.Graphics object should be re-created because of a bug related to updatePixels().
                    for (var update_g = 0; update_g < max_colors; update_g++) {
                        var new_g = s.createGraphics(canvas_size, canvas_size);
                        new_g.image(s.mask[update_g], 0, 0);
                        s.mask[update_g].remove();
                        s.mask[update_g] = new_g;
                    }
                }
            }
        }

        s.clear_canvas = function () {
            for (var i = 0; i < max_colors; i++) {
                s.mask[i].clear();
            }
            s.body = null;
        }

        s.updateImage = function (url) {
            s.body = s.loadImage(url);
        }
    }
}

function OriginalNameSpace() {
    return function (s) {
        s.setup = function () {
            s.pixelDensity(1);
            s.createCanvas(canvas_size, canvas_size);
            s.body = null;
            s.cursor(s.HAND);

            s.r_x = Array(max_colors).fill(0);
            s.r_y = Array(max_colors).fill(0);
            s.d_x = Array(max_colors).fill(0);
            s.d_y = Array(max_colors).fill(0);
            mousePressed_here = false;
        }


        s.draw = function () {
            s.background(255);
            s.noTint();
            if (s.body != null) {
                s.image(s.body, 0, 0, s.width, s.height);
            }
            s.tint(255, 127);

            for (var i = 0; i < max_colors; i++) {
                s.image(p5_input_reference.mask[i], s.r_x[i], s.r_y[i]);
            }

        }

        s.mousePressed = function (e) {
            s.d_x[palette_selected_index] = s.mouseX;
            s.d_y[palette_selected_index] = s.mouseY;

            if (s.mouseX <= s.width && s.mouseX >= 0 && s.mouseY <= s.height && s.mouseY >= 0) {
                s.mousePressed_here = true;
            }
        }

        s.mouseReleased = function (e) {
            s.mousePressed_here = false;
        }

        s.mouseDragged = function (e) {
            if (ui_uninitialized || s.mousePressed_here == false) return;
            if (s.mouseX <= s.width && s.mouseX >= 0 && s.mouseY <= s.height && s.mouseY >= 0) {

                s.r_x[palette_selected_index] += s.mouseX - s.d_x[palette_selected_index];
                s.r_y[palette_selected_index] += s.mouseY - s.d_y[palette_selected_index];

                s.d_x[palette_selected_index] = s.mouseX;
                s.d_y[palette_selected_index] = s.mouseY;
            }
        }

        s.updateImage = function (url) {
            s.body = s.loadImage(url);
        }


        s.clear_canvas = function () {
            s.body = null;

            for (var i = 0; i < max_colors; i++) {
                s.r_x[i] = 0;
                s.r_y[i] = 0;
                s.d_x[i] = 0;
                s.d_y[i] = 0;
            }

        }

    }
}

function generateOutputNameSpace() {
    return function (s) {
        s.setup = function () {
            s.pixelDensity(1);
            s.createCanvas(canvas_size, canvas_size);

            s.images = [];
            s.currentImage = 0;
            s.frameRate(15);
        }

        s.draw = function () {
            s.background(255);
            if (s.images.length > s.currentImage) {
                s.background(255);
                s.image(s.images[s.currentImage], 0, 0, s.width, s.height);
            }
        }

        s.updateImages = function (urls) {
            for (var i = urls.length - 1; i >= 0; i--) {
                var img = s.loadImage(urls[i]);
                s.images[i] = img;
            }
            s.currentImage = urls.length - 1;
        }

        s.changeCurrentImage = function (index) {
            if (index < s.images.length) {
                s.currentImage = index;
            }
        }

        s.clear_canvas = function () {
            s.images = [];
            s.currentImage = 0;
        }
    }
}

function updateResult() {
    disableUI();

    var canvas_reference = $('#p5-reference canvas').slice(1);
    var data_reference = [];

    for (var canvas_i = 0; canvas_i < max_colors; canvas_i++) {
        data_reference.push(canvas_reference[canvas_i].toDataURL('image/png').replace(/data:image\/png;base64,/, ''));
    }

    $.ajax({
        type: "POST",
        url: "/post",
        data: JSON.stringify({ "id": id, "original": original_image, "references": palette, "data_reference": data_reference, "shift_original": [p5_input_original.r_x, p5_input_original.r_y], "colors": colors }),
        dataType: "json",
        contentType: "application/json",
    }).done(function (data, textStatus, jqXHR) {

        let urls = data['result'];

        $('#ex1').slider({ 'max': urls.length - 1, "setValue": urls.length - 1 });
        p5_output.updateImages(urls);

        $("#ex1").attr('data-slider-value', urls.length - 1);
        $("#ex1").slider('refresh');

        enableUI();
    });
}

function enableUI() {
    ui_uninitialized = false;
    $("button").removeAttr('disabled');
    $('#ex1').slider('enable');
}

function disableUI() {
    ui_uninitialized = true;
    $("button").attr('disabled', true);
    $('#ex1').slider('disable');
}


$(function () {
    $("#main-ui-submit").click(function () {
        updateResult();
    });

    $("#sketch-clear").click(function () {
        p5_input_reference.clear_canvas();
        p5_input_original.clear_canvas();
        p5_output.clear_canvas()
        $('.palette-item-class').remove();
        palette = [];
        original_image = null;
        palette_selected_index = null;
        $("#palette-eraser").click();

        $("#sketch-clear").attr('disabled', true);
        $("#main-ui-submit").attr('disabled', true);
    });

    for (var i = 0; i < image_paths.length; i++) {
        var image_name = image_paths[i];

        $("#class-picker").append(
            '<option data-img-src="' + base_path + image_name + '" data-img-alt="' + image_name + '" value="' + (image_name) + '">' + image_name + '</option>'
        );
    }

    $("#class-picker").imagepicker({
        hide_select: false,
    });
    $('#class-picker').after(
        "<button type=\"submit\" class=\"form-control btn btn-primary col-md-2\" id=\"class-picker-submit-reference\">add to reference</button>"
    );
    $('#class-picker').after(
        "<button type=\"submit\" class=\"form-control btn btn-success col-md-2\" id=\"class-picker-submit-original\">add to original</button>"
    );

    $("#class-picker-submit-reference").after(
        "<div class=\"row\" id=\"class-picker-ui\"></div>"
    )
    $("#class-picker").appendTo("#class-picker-ui");
    $("#class-picker-submit-reference").appendTo("#class-picker-ui");
    $("#class-picker-submit-original").appendTo("#class-picker-ui");

    $("#class-picker-submit-reference").click(function () {
        const selected_class = $("#class-picker").val();
        const image_name_without_ext = selected_class.split('.').slice(0, -1).join('.');

        if (palette.length >= max_colors || palette.indexOf(selected_class) != -1) {
            return;
        }

        $("#palette-body").append(
            "<li class=\"palette-item palette-item-class\" id=\"palette-" + image_name_without_ext + "\"" +
            "data-class=\"" + selected_class +
            "\" style=\"background-color: " + colors[palette.length] + ";\"></li>");

        $("#palette-" + image_name_without_ext).append(
            "<img src=\"" + base_path + selected_class + "\">"
        )

        palette.push((selected_class));

        $("#palette-" + image_name_without_ext).click(function () {
            $(".palette-item.selected").removeClass('selected');
            $(this).addClass('selected');
            p5_input_reference.updateImage(base_path + selected_class);
            palette_selected_index = palette.indexOf(selected_class);

        });
        $("#palette-" + image_name_without_ext).click();
        palette_selected_index = palette.indexOf(selected_class);
        if (palette.length > 0 && original_image != null) {
            enableUI();
        }
    });

    $("#class-picker-submit-original").click(function () {
        selected_class = $("#class-picker").val();
        p5_input_original.updateImage(base_path + selected_class);
        original_image = selected_class;
        if (palette.length > 0 && original_image != null) {
            enableUI();
        }
    });

    $("#palette-eraser").click(function () {
        $(".palette-item.selected").removeClass('selected');
        $(this).addClass('selected');
    });

    $('#ex1').slider({
        formatter: function (value) {
            return 'interpolation: ' + (value / (16 - 1)).toFixed(2);
        }
    });
    $('#ex1').slider('disable');
    $("#ex1").change(function () {
        p5_output.changeCurrentImage(parseInt($("#ex1").val()));
    });

    p5_input_reference = new p5(ReferenceNameSpace(), "p5-reference");
    p5_input_original = new p5(OriginalNameSpace(), "p5-original");
    p5_output = new p5(generateOutputNameSpace(), "p5-right");

    // https://cofs.tistory.com/363
    var getCookie = function (name) {
        var value = document.cookie.match('(^|;) ?' + name + '=([^;]*)(;|$)');
        return value ? value[2] : null;
    };

    var setCookie = function (name, value, day) {
        var date = new Date();
        date.setTime(date.getTime() + day * 60 * 60 * 24 * 1000);
        document.cookie = name + '=' + value + ';expires=' + date.toUTCString() + ';path=/';
    };

    id = getCookie("id");

    if (id == null) {
        // https://stackoverflow.com/questions/1349404/generate-random-string-characters-in-javascript
        length = 20;
        var result = '';
        var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        var charactersLength = characters.length;
        for (var i = 0; i < length; i++) {
            result += characters.charAt(Math.floor(Math.random() * charactersLength));
        }
        id = result;
        setCookie("id", result, 1);
    }
})
