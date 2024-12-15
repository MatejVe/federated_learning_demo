const NUM_CLIENTS = 3;
const EPOCHS = 20;
const SAMPLE_SIZE = 5000; // Subset of MNIST to avoid memory crashes
const clientColors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
let mnistData;

let flChart, centralChart; // Chart instances

function initializeFLChart() {
    const flCtx = document.getElementById('flChart').getContext('2d');
    flChart = new Chart(flCtx, {
        type: 'line',
        data: { labels: Array.from({ length: EPOCHS }, (_, i) => i + 1), datasets: [] },
        options: { 
            plugins: { title: { display: true, text: 'Federated Learning: Client Metrics' } },
            scales: { y: { beginAtZero: true, max: 0.6 } },
            animation: false
        }
    });
}

function initializeCentralChart() {
    const centralCtx = document.getElementById('centralChart').getContext('2d');
    centralChart = new Chart(centralCtx, {
        type: 'line',
        data: { labels: Array.from({ length: EPOCHS }, (_, i) => i + 1), datasets: [] },
        options: { 
            plugins: { title: { display: true, text: 'Centralized Model Metrics' } },
            scales: { y: {beginAtZero: true, max: 0.6 } },
            animation: false
        }
    });
}

// Show and hide loading spinner
function showLoading(statusMessage) {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('status').innerText = statusMessage;
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

async function loadMNIST() {
    if (mnistData) return mnistData;

    const imageURL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
    const labelURL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

    const [imageBuffer, labelBuffer] = await Promise.all([
        fetch(imageURL).then(res => res.arrayBuffer()),
        fetch(labelURL).then(res => res.arrayBuffer())
    ]);

    const imageData = new Uint8Array(imageBuffer);
    const labelData = new Uint8Array(labelBuffer);

    const images = [];
    const labels = [];
    const testImages = [];
    const testLabels = [];

    // Split the dataset into training (5000) and test (1000)
    for (let i = 0; i < SAMPLE_SIZE; i++) {
        const image = [];
        for (let j = 0; j < 784; j++) {
            image.push(imageData[i * 784 + j] / 255);
        }

        if (i < SAMPLE_SIZE - 1000) {
            images.push(image);
            labels.push(labelData[i]);
        } else {
            testImages.push(image);
            testLabels.push(labelData[i]);
        }
    }

    mnistData = { xs: images, labels: labels, testXs: testImages, testLabels: testLabels };
    return mnistData;
}


async function splitDataForClients() {
    const data = await loadMNIST();
    const splitSize = Math.floor(data.xs.length / NUM_CLIENTS);

    return Array.from({ length: NUM_CLIENTS }, (_, i) => ({
        xs: data.xs.slice(i * splitSize, (i + 1) * splitSize),
        labels: data.labels.slice(i * splitSize, (i + 1) * splitSize)
    }));
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    return model;
}

async function trainClient(data, clientId) {
    const model = createModel();
    const xs = tf.tensor2d(data.xs);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), 10);

    await model.fit(xs, ys, {
        epochs: EPOCHS,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (!flChart.data.datasets[clientId]) {
                    flChart.data.datasets.push({ 
                        label: `Client ${clientId + 1} Loss`, 
                        data: [], 
                        borderWidth: 2 ,
                        borderColor: clientColors[clientId % clientColors.length],
                        fill: false
                    });
                }
                flChart.data.datasets[clientId].data.push(logs.loss);
                flChart.update();
            }
        }
    });

    const evalResult = await model.evaluate(xs, ys);
    const accuracy = evalResult[1].dataSync()[0];
    console.log(`Client ${clientId} Accuracy: ${accuracy}`);

    const weights = model.getWeights().map(w => w.clone());
    xs.dispose();
    ys.dispose();
    return { weights, accuracy };
}

function aggregateWeights(clientWeights) {
    const averagedWeights = clientWeights[0].weights.map(w => tf.zerosLike(w));
    clientWeights.forEach(({ weights }) => {
        weights.forEach((w, i) => averagedWeights[i] = averagedWeights[i].add(w));
    });
    return averagedWeights.map(w => w.div(tf.scalar(clientWeights.length)));
}

async function evaluateGlobalModel(aggregatedWeights) {
    const data = await loadMNIST();
    const testXs = tf.tensor2d(data.testXs);
    const testYs = tf.oneHot(tf.tensor1d(data.testLabels, 'int32'), 10);

    // Initialize a new model and set weights
    const globalModel = createModel();
    globalModel.setWeights(aggregatedWeights);

    // Evaluate the model on the test set
    const evalResult = await globalModel.evaluate(testXs, testYs);
    const testAccuracy = evalResult[1].dataSync()[0];

    document.getElementById('outputFL').innerText += `\nGlobal Model Test Accuracy: ${(testAccuracy * 100).toFixed(2)}%`;

    // Dispose of tensors
    testXs.dispose();
    testYs.dispose();
    globalModel.dispose();
}


async function startFederatedLearning() {
    showLoading("Federated Learning in Progress...");
    initializeFLChart();

    const clientsData = await splitDataForClients();
    let clientResults = [];

    for (let i = 0; i < NUM_CLIENTS; i++) {
        const result = await trainClient(clientsData[i], i);
        clientResults.push(result);
    }

    // Aggregate weights
    const aggregatedWeights = aggregateWeights(clientResults);

    const avgAccuracy = clientResults.reduce((sum, r) => sum + r.accuracy, 0) / NUM_CLIENTS;
    document.getElementById('outputFL').innerText = `Federated Learning Complete!\nAverage Client Accuracy: ${(avgAccuracy * 100).toFixed(2)}%`;

    // Evaluate the global model on the test set
    await evaluateGlobalModel(aggregatedWeights);
    hideLoading();
}


async function trainCentralizedModel() {
    showLoading("Centralized Learning in Progress...");
    initializeCentralChart();

    const data = await loadMNIST();
    const xs = tf.tensor2d(data.xs);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), 10);
    const testXs = tf.tensor2d(data.testXs);
    const testYs = tf.oneHot(tf.tensor1d(data.testLabels, 'int32'), 10);

    const model = createModel();

    await model.fit(xs, ys, {
        epochs: EPOCHS,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (!centralChart.data.datasets[0]) {
                    centralChart.data.datasets.push({ label: 'Centralized Loss', data: [], borderWidth: 2 });
                }
                centralChart.data.datasets[0].data.push(logs.loss);
                centralChart.update();
            }
        }
    });

    const evalResult = await model.evaluate(xs, ys);
    const accuracy = evalResult[1].dataSync()[0];
    const testEvalresult = await model.evaluate(testXs, testYs);
    const testAccuracy = testEvalresult[1].dataSync()[0];

    document.getElementById('outputCL').innerText = `Centralized Training Complete!\nAccuracy: ${(accuracy * 100).toFixed(2)}%\nTest Set Accuracy: ${(testAccuracy * 100).toFixed(2)}%`;
    xs.dispose();
    ys.dispose();
    testXs.dispose();
    testYs.dispose();
    hideLoading();
}
