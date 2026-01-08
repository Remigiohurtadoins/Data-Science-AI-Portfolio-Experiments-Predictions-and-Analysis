function previewImageInput() {
        let preview = document.getElementById('img_tag')
        let file = document.querySelector('input[type=file]').files[0];
        let reader = new FileReader();

        reader.addEventListener('load', function () {
            preview.src = reader.result;
        }, false);
        if (file) {
            reader.readAsDataURL(file);
        }
    }

    function getBase64(file) {
        var reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            sendData(reader.result);
        }
    }

    function sendData(data) {
        var token = document.getElementsByName('csrfmiddlewaretoken');
        let form = $('form')[0];
        $.ajax({
            type: 'POST',
            url: '/apiCNN/predecir/',
            data: {'mydata': data, 'csrfmiddlewaretoken': '{{ csrf_token }}'},
            success: function (data, textStatus) {
                json = JSON.stringify(data);
                json = JSON.parse(json);
                document.getElementById('clase').innerHTML = "<strong>Clase Predicha: </strong>" + json.clase
                document.getElementById('probabilidad').innerHTML = "<strong>Probabilidad: </strong>" + json.probabilidad + '%'
                document.getElementById('div_prediccion').style.display = 'block'
                document.getElementById("div_prediccion").scrollIntoView();
            },
            error: function (xhr, status, e) {
                alert(status, e);
            }
        });
    }

    function listarRegistros() {
        $.ajax({
            type: 'POST',
            url: '/apiCNN/listar/',
            data: {'csrfmiddlewaretoken': '{{ csrf_token }}'},
            success: function (data) {
                json = JSON.parse(data);
                let tag_str = "";
                document.getElementById('tabla').style.visibility = 'visible';
                for (let i = 0; i < json.length; i++) {
                    let obj = json[i];
                    if(obj.fields.image_name !== undefined) {
                        tag_str += "<tr>" +
                            "<td><img src=/media/" + obj.fields.image_name + " class='zoom_image'></td>" +
                            "<td>" + obj.fields.animal_predicted + "</td>" +
                            "<td>" + obj.fields.animal_percentage + "%</td>" +
                            "</tr>";
                    }
                }
                document.getElementById('table_content').innerHTML = tag_str;
                document.getElementById("tabla").scrollIntoView();
            }
        });

    }

    async function enviar() {
        let file = document.getElementById('elm_img').files[0];
        if (!requeried(document.getElementById('elm_img'))) {
            getBase64(file);
        } else {
            alert('Debe Seleccionar una Imagen');
        }
    }

    function requeried(inputtx) {
        return inputtx.value.length === 0;
    }