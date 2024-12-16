// Constants
const NUM_CLIENTS = 5;
const LOCAL_EPOCHS = 5;
const GLOBAL_EPOCHS = 5;
const CENTRALIZED_EPOCHS = 10;
const SAMPLE_SIZE = 5000; // Subset of MNIST to avoid memory crashes
const clientColors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];

// Global Variables
let mnistData;
let flChartLoss, flChartAccuracy, centralChart; // Chart instances

/******************************************************
 * 1. CHART INITIALIZATION FUNCTIONS
 ******************************************************/
function initializeFLChartLoss() {
    const flCtxLoss = document.getElementById('flChartLoss').getContext('2d');
    flChartLoss = new Chart(flCtxLoss, {
        type: 'line',
        data: { labels: Array.from({ length: LOCAL_EPOCHS * GLOBAL_EPOCHS }, (_, i) => i + 1), datasets: [] },
        options: {
            plugins: { title: { display: true, text: 'Federated Learning: Client Loss' } },
            scales: { y: { beginAtZero: true, max: 0.6, title: { display: true, text: 'Loss' }} , 
                    x: {title: {display: true, text: 'Local Epoch'}}},
            animation: false
        }
    });
}

function initializeFLChartAccuracy() {
    const flCtxAccuracy = document.getElementById('flChartAccuracy').getContext('2d');
    flChartAccuracy = new Chart(flCtxAccuracy, {
        type: 'line',
        data: { labels: Array.from({ length: GLOBAL_EPOCHS }, (_, i) => i + 1), datasets: [] },
        options: {
            plugins: { title: { display: true, text: 'Federated Learning: Accuracy' } },
            scales: { y: { title: {display: true, text: 'Accuracy'}}, x: { title: {display: true, text: 'Global Epoch'}}},
            animation: false
        }
    });
}

function initializeCentralChart() {
    const centralCtx = document.getElementById('centralChart').getContext('2d');
    centralChart = new Chart(centralCtx, {
        type: 'line',
        data: { labels: Array.from({ length: CENTRALIZED_EPOCHS }, (_, i) => i + 1), datasets: [] },
        options: {
            plugins: { title: { display: true, text: 'Centralized Model Metrics' } },
            scales: { 
                y: { 
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Loss' },
                    beginAtZero: true 
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: {display: true, text: 'Accuracy'},
                    grid: {drawOnChartArea: false}, // Prevent grid overlap
                    max: 1
                }
            },
            animation: false
        }
    });
}

/******************************************************
 * 2. SPINNER CONTROL FUNCTIONS
 ******************************************************/
function showLoading(statusMessage) {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('status').innerText = statusMessage;
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

/******************************************************
 * 3. MNIST DATA LOADING FUNCTIONS
 ******************************************************/
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

    const images = [], labels = [], testImages = [], testLabels = [];

    for (let i = 0; i < SAMPLE_SIZE; i++) {
        const image = [];
        for (let j = 0; j < 784; j++) image.push(imageData[i * 784 + j] / 255);

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

/******************************************************
 * 4. MODEL CREATION AND AGGREGATION
 ******************************************************/
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    return model;
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

    const globalModel = createModel();
    globalModel.setWeights(aggregatedWeights);

    const evalResult = await globalModel.evaluate(testXs, testYs);
    const testAccuracy = evalResult[1].dataSync()[0];

    testXs.dispose();
    testYs.dispose();
    globalModel.dispose();

    return testAccuracy;
}

/******************************************************
 * 5. FEDERATED LEARNING FUNCTIONS
 ******************************************************/
async function trainClient(data, clientId, weights) {
    const model = createModel();
    model.setWeights(weights);

    const xs = tf.tensor2d(data.xs);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), 10);

    await model.fit(xs, ys, {
        epochs: LOCAL_EPOCHS,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (!flChartLoss.data.datasets[clientId]) {
                    flChartLoss.data.datasets.push({ 
                        label: `Client ${clientId + 1} Loss`,
                        data: [], borderColor: clientColors[clientId % clientColors.length], fill: false
                    });
                }
                flChartLoss.data.datasets[clientId].data.push(logs.loss);
                flChartLoss.update();
            }
        }
    });

    const accuracy = (await model.evaluate(xs, ys))[1].dataSync()[0];
    const updatedWeights = model.getWeights().map(w => w.clone());

    xs.dispose();
    ys.dispose();
    return { weights: updatedWeights, accuracy };
}

async function startFederatedLearning() {
    showLoading("Federated Learning in Progress...");
    initializeFLChartLoss();
    initializeFLChartAccuracy();

    const model = createModel();
    let weights = model.getWeights().map(w => w.clone());
    const clientsData = await splitDataForClients();

    // Initialize datasets if not already present
    if (flChartAccuracy.data.datasets.length === 0) {
        flChartAccuracy.data.datasets.push({
            label: 'Global Test Set Accuracy',
            data: [],
            borderColor: '#36A2EB', // Blue
            yAxisID: 'y1',
            fill: false
        });
        flChartAccuracy.data.datasets.push({
            label: 'Average Train Accuracy',
            data: [],
            borderColor: '#FF9F40', // Orange
            yAxisID: 'y1',
            fill: false
        });
    }

    for (let i = 0; i < GLOBAL_EPOCHS; i++) {
        let epochResults = [];
        let trainAccuracySum = 0;

        // Train each client
        for (let j = 0; j < NUM_CLIENTS; j++) {
            const result = await trainClient(clientsData[j], j, weights);
            epochResults.push(result);
            trainAccuracySum += result.accuracy; // Sum up client accuracies
        }

        // Aggregate client weights
        weights = aggregateWeights(epochResults);

        // Calculate Average Train Accuracy
        const avgTrainAccuracy = trainAccuracySum / NUM_CLIENTS;

        // Evaluate on the global test set
        const testAccuracy = await evaluateGlobalModel(weights);

        // Update chart datasets
        flChartAccuracy.data.datasets[0].data.push(testAccuracy);    // Global Test Set Accuracy
        flChartAccuracy.data.datasets[1].data.push(avgTrainAccuracy); // Average Train Accuracy

        flChartAccuracy.update();
    }

    document.getElementById('outputFL').innerText = "Federated Learning Complete!";
    hideLoading();
}


/******************************************************
 * 6. CENTRALIZED LEARNING FUNCTION
 ******************************************************/
async function trainCentralizedModel() {
    showLoading("Centralized Learning in Progress...");
    initializeCentralChart();

    const data = await loadMNIST();
    const xs = tf.tensor2d(data.xs);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), 10);
    const testXs = tf.tensor2d(data.testXs);
    const testYs = tf.oneHot(tf.tensor1d(data.testLabels, 'int32'), 10);

    const model = createModel();

    // Initialize datasets if not already present
    if (centralChart.data.datasets.length === 0) {
        centralChart.data.datasets.push({
            label: 'Centralized Loss',
            data: [],
            borderColor: '#FF6384',
            yAxisID: 'y',
            fill: false
        });
        centralChart.data.datasets.push({
            label: 'Training Accuracy',
            data: [],
            borderColor: '#36A2EB',
            yAxisID: 'y1',
            fill: false
        });
        centralChart.data.datasets.push({
            label: 'Test Accuracy',
            data: [],
            borderColor: '#4BC0C0',
            yAxisID: 'y1',
            fill: false
        });
    }

    // Train the model and update chart
    await model.fit(xs, ys, {
        epochs: CENTRALIZED_EPOCHS,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Evaluate on test set
                const testEval = await model.evaluate(testXs, testYs);
                const testAccuracy = testEval[1].dataSync()[0];

                // Update datasets
                centralChart.data.datasets[0].data.push(logs.loss);      // Loss
                centralChart.data.datasets[1].data.push(logs.acc || 0); // Training Accuracy
                centralChart.data.datasets[2].data.push(testAccuracy);  // Test Accuracy

                centralChart.update();
            }
        }
    });

    const finalTestEval = await model.evaluate(testXs, testYs);
    const finalTestAccuracy = finalTestEval[1].dataSync()[0];

    document.getElementById('outputCL').innerText = 
        `Centralized Training Complete!\nFinal Test Set Accuracy: ${(finalTestAccuracy * 100).toFixed(2)}%`;

    xs.dispose();
    ys.dispose();
    testXs.dispose();
    testYs.dispose();
    hideLoading();
}
