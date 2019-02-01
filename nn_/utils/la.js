(function() {
  function selectLayer(layer, layer_el) {
    if (active_layer) {
      deselectLayer();
    }

    layer_el.classList.add('table-info');

    active_layer = layer;

    layer_index_el.innerHTML = `Layer ${layer.index + 1} Properties`;
    insert_el.innerHTML = 'Duplicate Layer';

    kernel_size_el.value = layer.kernel_size;
    stride_el.value = layer.stride;
    dilation_el.value = layer.dilation;
    padding_el.value = layer.padding_type;

    delete_el.removeAttribute('disabled');
  }

  function renderLayer(layer) {
    let layer_el = document.createElement('tr');

    if (layer === active_layer) {
      layer_el.classList.add('table-info');
    }

    layer_el.innerHTML = layer_template(layer.toJSON());
    layer_el.addEventListener('click', e => {
      e.stopPropagation();
      selectLayer(layer, layer_el);
    });

    layers_el.append(layer_el);
  }

  function insertLayer() {
    let kernel_size = parseFloat(kernel_size_el.value, 10);
    let stride = parseFloat(stride_el.value, 10);
    let dilation = parseFloat(dilation_el.value, 10);
    let padding_type = padding_el.value;
    let prev_layer = _.last(layers);
    let input_size = prev_layer ? prev_layer.output_size : parseFloat(input_size_el.value, 10);
    let index = layers.length;
    let layer = new Layer(index, input_size, kernel_size, stride, dilation, padding_type, prev_layer);
    layers.push(layer);

    renderLayer(layer);
    selectLayer(layer, layers_el.lastChild);
    updateChart();
  }

  function resetLayers() {
    layers_el.innerHTML = '';
    layers = [];
    updateChart();
  }

  function deleteLayer(layer) {
    deselectLayer();
    layers = _.without(layers, layer);
    updateLayers();
  }

  function updateChart() {
    chart.data.labels = layers.map(layer => `Layer ${layer.index + 1}`);
    chart.data.datasets[0].data = layers.map(layer => layer.receptive_field);
    chart.update();

    let hash = layers.map(layer => {
      return [
        layer.kernel_size,
        layer.stride,
        layer.dilation,
        layer.padding_type
      ].join(',');
    }).join(';');

    history.pushState(hash, document.title, `${window.location.pathname}#${hash}`);
  }

  function parseLayers() {
    let hash = window.location.hash.slice(1);
    if (hash === '') {
      return;
    }

    layers = hash.split(';').reduce((layers, layer_desc, index) => {
      let prev_layer = _.last(layers);
      let input_size = prev_layer ? prev_layer.output_size : parseFloat(input_size_el.value, 10);
      let [kernel_size, stride, dilation, padding_type] = layer_desc.split(',');
      let layer = new Layer(index, input_size, kernel_size, stride, dilation, padding_type, prev_layer);
      layers.push(layer);
      return layers;
    }, layers);

    updateLayers();
  }

  function updateLayers() {
    for (let i = 0; i < layers.length; i++) {
      let layer = layers[i];
      layer.index = i;

      if (i > 0) {
        let prev_layer = layers[i - 1];
        layer.prev_layer = prev_layer;
        layer.input_size = prev_layer.output_size;
      } else {
        layer.prev_layer = undefined;
        layer.input_size = parseFloat(input_size_el.value, 10);
      }
    }

    layers_el.innerHTML = '';
    layers.forEach(layer => renderLayer(layer));

    updateChart();
  }

  function deselectLayer() {
    if (!active_layer) {
      // only clear properties bar if it's empty
      return;
    }

    let child_els = layers_el.children;
    for (var i = 0; i < child_els.length; i++) {
      child_els[i].classList.remove('table-info');
    }

    active_layer = null;
    delete_el.setAttribute('disabled', true);
    layer_index_el.innerHTML = 'New Layer Properties';
    insert_el.innerHTML = 'Add Layer';
    kernel_size_el.value = '3';
    stride_el.value = '1';
    dilation_el.value = '1';
    padding_el.value = 'VALID';
  }

  let layers = [];

  let template_html = document.getElementById('layer-template').innerHTML;
  let layer_template = _.template(template_html.split("&lt;").join("<").split("&gt;").join(">"));

  let input_size_el = document.getElementById('input-size');
  let sidebar_el = document.getElementById('sidebar');
  let layers_el = document.getElementById('layers');

  let insert_el = document.getElementById('insert');
  let reset_el = document.getElementById('reset');
  let delete_el = document.getElementById('delete');

  let layer_index_el = document.getElementById('layer-index');
  let kernel_size_el = document.getElementById('kernel-size');
  let stride_el = document.getElementById('stride');
  let dilation_el = document.getElementById('dilation');
  let padding_el = document.getElementById('padding');

  let active_layer = null;

  window.addEventListener('click', e => {
    deselectLayer();
  });

  input_size_el.addEventListener('keyup', e => {
    if (layers.length == 0) {
      return;
    }
    updateLayers();
  });

  kernel_size_el.addEventListener('keyup', e => {
    if (active_layer != null) {
      active_layer.kernel_size = parseFloat(kernel_size_el.value, 10);
      updateLayers();
    }
  });

  stride_el.addEventListener('keyup', e => {
    if (active_layer != null) {
      active_layer.stride = parseFloat(stride_el.value, 10);
      updateLayers();
    }
  });

  dilation_el.addEventListener('keyup', e => {
    if (active_layer != null) {
      active_layer.dilation = parseFloat(dilation_el.value, 10);
      updateLayers();
    }
  });

  padding_el.addEventListener('change', e => {
    if (active_layer != null) {
      active_layer.padding_type = padding_el.value;
      updateLayers();
    }
  });

  sidebar_el.addEventListener('click', e => {
    // To prevent window click event and deselection when changing properties.
    e.stopPropagation();
  });

  insert_el.addEventListener('click', e => {
    e.preventDefault();
    insertLayer();
  });

  reset_el.addEventListener('click', e => {
    e.preventDefault();
    resetLayers();
  });

  delete_el.addEventListener('click', e => {
    e.preventDefault();
    if (active_layer) {
      deleteLayer(active_layer);
    }
  });

  let ctx = document.getElementById('growth').getContext('2d');
  let chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
          label: 'Receptive Field Growth',
          lineTension: 0,
          data: [],
          backgroundColor: ['rgba(88, 198, 194, 0.7)'],
          borderColor: ['rgba(39, 145, 140, 0.7)']
      }]
    },
    options: {
      maintainAspectRatio: false,
      legend: {
        display: true,
        labels: {
            fontSize: 16
        }
      }
    }
  });

  parseLayers();

}).call(this);
