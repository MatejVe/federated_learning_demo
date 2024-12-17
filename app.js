// Constants
const NUM_CLIENTS = 3;
const LOCAL_EPOCHS = 5;
const GLOBAL_EPOCHS = 5;
const CENTRALIZED_EPOCHS = 10;
const FEDOPT_LOCAL_EPOCHS = 5;
const FEDOPT_GLOBAL_EPOCHS = 10;
const SAMPLE_SIZE = 5000; // Subset of cifar to avoid memory crashes
const TEST_SIZE = 1000;
const NUM_FEATURES = 10;
const clientColors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];

// Global Variables
let data;
let flChartLoss, flChartAccuracy, centralChart, proxChartLoss, proxChartAccuracy; // Chart instances

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
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Loss' }} , 
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
            scales: { x: { title: {display: true, text: 'Global Epoch'}}},
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

function initializeFedOptChartLoss() {
    const fedOptCtxLoss = document.getElementById('fedOptChartLoss').getContext('2d');
    fedOptChartLoss = new Chart(fedOptCtxLoss, {
        type: 'line',
        data: { labels: Array.from({length: FEDOPT_LOCAL_EPOCHS * FEDOPT_GLOBAL_EPOCHS}, (_, i) => i + 1), datasets: []},
        options: {
            plugins: { title: { display: true, text: 'FedProx: Client Loss' }},
            scales: { x: {title: {display: true, text: 'Local Epoch'}},
                    y: { beginAtZero: true, title: { display: true, text: 'Loss'}}},
            animation: false
        }
    });
}

function initializeFedOptChartAccuracy() {
    const fedOptCtxAccuracy = document.getElementById('fedOptChartAccuracy').getContext('2d');
    fedOptChartAccuracy = new Chart(fedOptCtxAccuracy, {
        type: 'line',
        data: {labels: Array.from({length: FEDOPT_GLOBAL_EPOCHS}, (_, i) => i + 1), datasets: []},
        options: {
            plugins: {title: {display: true, text: 'FedProx: Accuracy'}},
            scales: {x: {title: {display: true, text: 'Global Epoch'}}},
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
 * 3. CIFAR DATA LOADING FUNCTIONS
 ******************************************************/
function generateSyntheticDataWithTest(numSamples = SAMPLE_SIZE, testSize = TEST_SIZE, numFeatures = NUM_FEATURES, numClients = NUM_CLIENTS) {
    if (data) return data;
    
    console.log("Generating structured synthetic dataset...");

    const totalSamplesPerClient = Math.floor(numSamples / numClients);

    // Define label distributions for each client
    const clientLabelDistribution = [
        [2, 3, 4, 5, 6, 7, 8], 
        [0, 1, 2, 3, 4],    // Client 1 gets labels 0-3
              // Client 2 gets labels 4-6
        [5, 6, 7, 8, 9]        // Client 3 gets labels 7-9
    ];

    const clientData = Array.from({ length: numClients }, () => ({ xs: [], labels: [] }));
    const testData = { xs: [], labels: [] };

    function generateFeaturesForLabel(label) {
        // Use Gaussian distributions with mean dependent on label
        const mean = label; // Shift means for each label
        return Array.from({ length: numFeatures }, () => mean + Math.random() * 0.5 - 0.5);
    }

    // Generate client data
    for (let clientId = 0; clientId < numClients; clientId++) {
        const allowedLabels = clientLabelDistribution[clientId];
        for (let i = 0; i < totalSamplesPerClient; i++) {
            const label = allowedLabels[Math.floor(Math.random() * allowedLabels.length)];
            const features = generateFeaturesForLabel(label);
            clientData[clientId].xs.push(features);
            clientData[clientId].labels.push(label);
        }
    }

    // Generate test data (uniformly across all labels)
    for (let i = 0; i < testSize; i++) {
        const label = Math.floor(Math.random() * 10); // All 10 labels
        const features = generateFeaturesForLabel(label);
        testData.xs.push(features);
        testData.labels.push(label);
    }

    data = {
        clientData,
        testData
    };

    console.log("Structured synthetic dataset generated.");
    return data;
}

/******************************************************
 * 4. MODEL CREATION AND AGGREGATION
 ******************************************************/
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [10] })); // Input size: 10
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' })); // 10 classes
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
    const testXs = tf.tensor2d(data.testData.xs);
    const testYs = tf.oneHot(tf.tensor1d(data.testData.labels, 'int32'), 10);

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

    // Explicitly reshape the input data
    const inputShape = [data.xs.length, NUM_FEATURES];
    const xs = tf.tensor2d(data.xs, inputShape);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), 10);

    await model.fit(xs, ys, {
        epochs: LOCAL_EPOCHS,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (!flChartLoss.data.datasets[clientId]) {
                    flChartLoss.data.datasets.push({
                        label: `Client ${clientId + 1} Loss`,
                        data: [],
                        borderWidth: 2,
                        borderColor: clientColors[clientId % clientColors.length],
                        fill: false
                    });
                }
                flChartLoss.data.datasets[clientId].data.push(logs.loss);
                flChartLoss.update();
            }
        }
    });

    // Evaluate the model to get accuracy
    const evalResult = await model.evaluate(xs, ys);
    const accuracy = evalResult.length > 1 ? evalResult[1].dataSync()[0] : 0; // Check for accuracy tensor
    console.log(`Client ${clientId} Accuracy: ${accuracy}`);

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

    const allData = await generateSyntheticDataWithTest();
    const clientsData = allData.clientData;

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
    const finalAccuracy = await evaluateGlobalModel(weights);

    document.getElementById('outputFL').innerText = `Federated Learning Complete!\nFinal Test Set Accuracy: ${(finalAccuracy * 100).toFixed(2)}%`;
    hideLoading();
}

async function trainClientFedOpt(data, clientId, globalWeights) {
    const model = createModel();
    model.setWeights(globalWeights); // Initialize with global weights

    const inputShape = [data.xs.length, 10];
    const xs = tf.tensor2d(data.xs, inputShape);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), 10);

    await model.fit(xs, ys, {
        epochs: FEDOPT_LOCAL_EPOCHS,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (!fedOptChartLoss.data.datasets[clientId]) {
                    fedOptChartLoss.data.datasets.push({
                        label: `Client ${clientId + 1} Loss`,
                        data: [],
                        borderWidth: 2,
                        borderColor: clientColors[clientId % clientColors.length],
                        fill: false
                    });
                }
                fedOptChartLoss.data.datasets[clientId].data.push(logs.loss);
                fedOptChartLoss.update();
            }
        }
    });
    // Evaluate the model to get accuracy
    const evalResult = await model.evaluate(xs, ys);
    const accuracy = evalResult.length > 1 ? evalResult[1].dataSync()[0] : 0; // Check for accuracy tensor
    console.log(`Client ${clientId} Accuracy: ${accuracy}`);

    // Compute weight deltas (local model - global model)
    const updatedWeights = model.getWeights();
    const weightDeltas = updatedWeights.map((w, i) => w.sub(globalWeights[i]));

    // Dispose of tensors
    xs.dispose();
    ys.dispose();

    return { weightDeltas, accuracy };
}

function aggregateWeightsFedOpt(globalWeights, clientUpdates, learningRate = 0.1) {
    // Iterate over client updates and apply SGD sequentially
    clientUpdates.forEach(({ weightDeltas }) => {
        globalWeights = globalWeights.map((w, i) => 
            w.sub(weightDeltas[i].mul(tf.scalar(learningRate)))
        );
    });

    return globalWeights;
}


async function startFedOptLearning() {
    showLoading("FedOpt in Progress...");
    initializeFedOptChartLoss();
    initializeFedOptChartAccuracy();

    const model = createModel();
    let globalWeights = model.getWeights().map(w => w.clone());

    const allData = await generateSyntheticDataWithTest();
    const clientsData = allData.clientData;

    // Initialize datasets if not already present
    if (fedOptChartAccuracy.data.datasets.length === 0) {
        fedOptChartAccuracy.data.datasets.push({
            label: 'Global Test Set Accuracy',
            data: [],
            borderColor: '#36A2EB', // Blue
            yAxisID: 'y1',
            fill: false
        });
        fedOptChartAccuracy.data.datasets.push({
            label: 'Average Train Accuracy',
            data: [],
            borderColor: '#FF9F40', // Orange
            yAxisID: 'y1',
            fill: false
        });
    }

    for (let epoch = 0; epoch < FEDOPT_GLOBAL_EPOCHS; epoch++) {
        let clientUpdates = [];
        let trainAccuracySum = 0;

        // Train clients and collect weight deltas
        for (let j = 0; j < NUM_CLIENTS; j++) {
            const result = await trainClientFedOpt(clientsData[j], j, globalWeights);
            clientUpdates.push(result);
            trainAccuracySum += result.accuracy;
        }

        // Aggregate updates using server-side Adam
        globalWeights = aggregateWeightsFedOpt(globalWeights, clientUpdates);
        
        const avgTrainAccuracy = trainAccuracySum / NUM_CLIENTS;

        // Evaluate global model on the test set
        const testAccuracy = await evaluateGlobalModel(globalWeights);
        console.log(`Global Epoch ${epoch + 1}: Test Accuracy = ${testAccuracy}`);

        // Update chart
        fedOptChartAccuracy.data.datasets[0].data.push(testAccuracy);
        fedOptChartAccuracy.data.datasets[1].data.push(avgTrainAccuracy);
        fedOptChartAccuracy.update();
    }
    const finalAccuracy = await evaluateGlobalModel(globalWeights);

    document.getElementById('outputFedOpt').innerText = `FedOpt Complete!\nFinal Test Set Accuracy: ${(finalAccuracy * 100).toFixed(2)}%`;
    hideLoading();
}

/******************************************************
 * 6. CENTRALIZED LEARNING FUNCTION
 ******************************************************/
async function trainCentralizedModel() {
    showLoading("Centralized Learning in Progress...");
    initializeCentralChart();

    const data = await generateSyntheticDataWithTest();
    const allXs = data.clientData.flatMap(client => client.xs);
    const allLabels = data.clientData.flatMap(client => client.labels);
    const xs = tf.tensor2d(allXs);
    const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), 10);
    const testXs = tf.tensor2d(data.testData.xs);
    const testYs = tf.oneHot(tf.tensor1d(data.testData.labels, 'int32'), 10);

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
