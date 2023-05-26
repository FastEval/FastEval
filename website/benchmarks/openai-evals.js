import { round, allowCharacterLineBreaks, computeUpdatedHash } from '../utils.js'
import { createConversationItemE } from '../components/conversation-item.js'
import { createLinkE } from '../components/link.js'
import { createModelSelectV } from '../components/model-select.js'
import { createTextE } from '../components/text.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

function getScores(spec, finalReport) {
    switch (spec.run_config.eval_spec.cls) {
        case 'evals.elsuite.modelgraded.classify:ModelBasedClassify':
            if (finalReport.score && finalReport.metascore)
                return round(finalReport.score) + ' | ' + round(finalReport.metascore)
            else if (finalReport.score)
                return round(finalReport.score)
            return null
        case 'evals.elsuite.basic.match:Match':
            return round(finalReport.accuracy)
        case 'evals.elsuite.basic.fuzzy_match:FuzzyMatch':
            return round(finalReport.f1_score)
        case 'evals.elsuite.basic.includes:Includes':
            return round(finalReport.accuracy)
        case 'evals.elsuite.multiple_choice:MultipleChoice':
            return round(finalReport.accuracy)
        case 'evals.elsuite.translate:Translate':
            return round(finalReport.sacrebleu_score)
        default:
            throw new Error()
    }
}

function createConversationE(conversation) {
    if (typeof conversation === 'string')
        conversation = [{ role: 'user', content: conversation }]

    const containerE = document.createElement('div')

    for (const conversationItem of conversation) {
        let role
        if (conversationItem.role === 'system' && conversationItem.name === undefined)
            role = 'system'
        else if (conversationItem.role === 'system' && conversationItem.name === 'example_user')
            role = 'user'
        else if (conversationItem.role === 'system' && conversationItem.name === 'example_assistant')
            role = 'assistant'
        else if (conversationItem.role === 'user')
            role = 'user'
        else if (conversationItem.role === 'assistant')
            role = 'assistant'

        containerE.appendChild(createConversationItemE(role, conversationItem.content))
    }

    return containerE
}

function showDataFromModelBasedClassify(finalReport, samples) {
    return [Object.entries(finalReport).map(e => e[0] + ': ' + round(e[1])).sort(), [...samples.entries()].map(([sampleId, sample]) => [sampleId, [
        ...(() => {
            const usesMultipleInputs = sample.sampling.prompt.input1 !== undefined

            const ret = []
            function add(prompt, completion, i) {
                const indexIndictator =  (usesMultipleInputs ? ' [' + i + ']' : '') + ':'
                ret.push(createTextE('The model got the following start of a conversation as input' + indexIndictator))
                ret.push(createConversationE(prompt))
                ret.push(createTextE('The model answered in the following way:' + indexIndictator))
                ret.push(createConversationE([{ role: 'assistant', content: completion }]))
            }

            if (usesMultipleInputs) {
                for (let i = 1; i <= Math.max(Object.keys(sample.sampling.prompt).length, Object.keys(sample.sampling.sampled).length); i++)
                    add(sample.sampling.prompt['input' + i], sample.sampling.prompt['completion' + i] ?? sample.sampling.sampled['completion' + i], i)
            } else {
                add(sample.sampling.prompt.input, sample.sampling.prompt.completion ?? sample.sampling.sampled.completion)
            }

            return [
                ...ret,
                createTextE('The model was then asked to evaluate its own answer' + (usesMultipleInputs ? 's' : '') + ':'),
            ]
        })(),

        createConversationE(sample.sampling.info.prompt),

        createTextE('The model responded to this evaluation as follows:'),
        createConversationE([{ role: 'assistant', content: sample.sampling.info.sampled }]),

        createTextE('This was ' + (sample.sampling.info.invalid_choice ? 'an invalid' : 'a valid') + ' response to the evaluation.'
            + (sample.sampling.info.score !== null ? (' The resulting score is ' + sample.sampling.info.score + '.') : '')),
    ]])]
}

function createAnswersE(options) {
    if (typeof options === 'string')
        options = [options]
    const onlySingleOption = options.length === 1

    return [
        createTextE('The following answer' + (onlySingleOption ? ' was' : 's were') + ' expected:'),
        ...options.map(e => createConversationE([{ role: 'assistant', content: e }])),
    ]
}

function createMatchEs(prompt, correctAnswers, sampledModelAnswers, modelAnswerIsCorrect) {
    return [
        createTextE('The model got the following start of a conversation as input:'),
        createConversationE(prompt),

        ...createAnswersE(correctAnswers),

        createTextE('The model answered in the following way:'),
        createConversationE([{ role: 'assistant', content: sampledModelAnswers }]),

        createTextE('The model answer was judged to be ' + (modelAnswerIsCorrect ? 'correct.' : 'incorrect.')),
    ]
}

function showDataFromMatch(finalReport, samples) {
    return [
        ['Accuracy: ' + round(finalReport.accuracy)],
        [...samples.entries()].map(([sampleId, sample]) => [
            sampleId,
            createMatchEs(sample.match.prompt, sample.match.options, sample.match.sampled, sample.match.correct),
        ]),
    ]
}

function showDataFromFuzzyMatch(finalReport, samples) {
    return [
        [
            'Accuracy: ' + round(finalReport.accuracy),
            'F1 score: ' + round(finalReport.f1_score),
        ],
        [...samples.entries()].map(([sampleId, sample]) => [
            sampleId,
            createMatchEs(sample.match.test_sample.input, sample.match.test_sample.ideal, sample.match.sampled, sample.match.correct),
        ]),
    ]
}

function showDataFromIncludes(finalReport, samples) {
    return [
        ['Accuracy: ' + round(finalReport.accuracy)],
        [...samples.entries()].map(([sampleId, sample]) => [
            sampleId,
            createMatchEs(sample.match.prompt, sample.match.expected, sample.match.sampled, sample.match.correct),
        ]),
    ]
}

function showDataFromMultipleChoice(finalReport, samples) {
    return [
        ['Accuracy: ' + round(finalReport.accuracy)],
        [...samples.entries()].map(([sampleId, sample]) => [
            sampleId,
            createMatchEs(sample.match.prompt, sample.match.options, sample.match.sampled, sample.match.correct),
        ]),
    ]
}

function showDataFromTranslate(finalReport, samples) {
    return [
        ['SacreBLEU score: ' + round(finalReport.sacrebleu_score)],
        [...samples.entries()].map(([sampleId, sample]) => [
            sampleId,
            [
                ...createMatchEs(sample.match.prompt, sample.match.expected, sample.match.sampled, sample.match.correct),
                createTextE('The SacreBLEU score is ' + round(sample.metrics.sacrebleu_sentence_score) + '.'),
            ]
        ]),
    ]
}

function showData(spec, finalReport, samples) {
    switch (spec.run_config.eval_spec.cls) {
        case 'evals.elsuite.modelgraded.classify:ModelBasedClassify':
            return showDataFromModelBasedClassify(finalReport, samples)
        case 'evals.elsuite.basic.match:Match':
            return showDataFromMatch(finalReport, samples)
        case 'evals.elsuite.basic.fuzzy_match:FuzzyMatch':
            return showDataFromFuzzyMatch(finalReport, samples)
        case 'evals.elsuite.basic.includes:Includes':
            return showDataFromIncludes(finalReport, samples)
        case 'evals.elsuite.multiple_choice:MultipleChoice':
            return showDataFromMultipleChoice(finalReport, samples)
        case 'evals.elsuite.translate:Translate':
            return showDataFromTranslate(finalReport, samples)
        default:
            throw new Error()
    }
}

function createSamplesV(mappedSamples) {
    const containerE = document.createElement('div')
    containerE.classList.add('samples')

    for (const [sampleId, mappedSample] of mappedSamples) {
        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        containerE.appendChild(sampleE)

        const sampleIdE = createLinkE('ID: ' + sampleId, { sample: sampleId })

        sampleE.append(
            sampleIdE,
            ...mappedSample,
        )
    }

    return containerE
}

function createSelectedModelReportV(report, selectedSampleId) {
    const reportLines = report.split('\n')
    const [specLine, finalReportLine, ...dataLines] = reportLines
    const spec = JSON.parse(specLine).spec
    const finalReport = JSON.parse(finalReportLine).final_report
    const data = dataLines.filter(dataLine => dataLine.length !== 0).map(dataLine => JSON.parse(dataLine))

    const reportDataBySampleId = new Map()
    for (const event of data) {
        const item = reportDataBySampleId.get(event.sample_id) ?? {}
        item[event.type] = event.data
        reportDataBySampleId.set(event.sample_id, item)
    }

    const [finalReportLines, mappedSamples] = showData(spec, finalReport, new Map([...reportDataBySampleId.entries()]
        .sort(([k1, v1], [k2, v2]) => parseInt(k1.split('.').slice(-1)) - parseInt(k2.split('.').slice(-1)))))

    const containerE = document.createElement('div')
    containerE.classList.add('selected-model-report')

    const selectedModelInformationE = document.createElement('div')
    selectedModelInformationE.classList.add('selected-model-information')
    containerE.appendChild(selectedModelInformationE)

    if (!selectedSampleId)
        selectedModelInformationE.append(...finalReportLines.map(line => createTextE(line)))

    const samplesE = selectedSampleId
        ? createSamplesV(mappedSamples.filter(s => s[0] === selectedSampleId))
        : createSamplesV(mappedSamples)
    containerE.appendChild(samplesE)

    return containerE
}

async function createEvalReportsV(baseUrl, evalName, modelName, sampleId) {
    const modelNames = (await (await fetch(baseUrl + '/__index__.json')).json()).map(m => m[1])

    const containerE = document.createElement('div')

    const reportUrl = baseUrl + '/openai-evals/' + modelName.replace('/', '--') + '/' + evalName + '.json'
    const report = await (await fetch(reportUrl)).text()
    const spec = JSON.parse(report.split('\n')[0]).spec

    const finalReportInformationE = document.createElement('div')
    containerE.appendChild(finalReportInformationE)
    finalReportInformationE.classList.add('final-report-information')
    finalReportInformationE.appendChild(createTextE('Name: ' + spec.eval_name))
    finalReportInformationE.appendChild(createTextE('Evaluation method: ' + spec.run_config.eval_spec.cls.split(':').slice(-1)))

    const { view: modelSelectV, element: modelSelectE } = createModelSelectV('Model:', modelNames)
    containerE.appendChild(modelSelectV)

    modelSelectE.value = modelName.replace('/', '--')

    containerE.appendChild(createSelectedModelReportV(report, sampleId))

    modelSelectE.addEventListener('change', () => {
        location.hash = computeUpdatedHash({ model: modelSelectE.value.replace('--', '/') })
    })

    return containerE
}

export async function createEvalsIndexV(baseUrl) {
    const modelNames = (await (await fetch(baseUrl + '/__index__.json')).json()).map(m => m[1])

    const reportsIndexE = document.createElement('table')
    const tableHeadE = reportsIndexE.createTHead().insertRow()
    const tableBodyE = reportsIndexE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Eval name'))

    for (const modelName of modelNames)
        tableHeadE.insertCell().appendChild(createTextE(allowCharacterLineBreaks(modelName)))

    const reportsIndex = Object.fromEntries(await Promise.all(modelNames.map(async modelName =>
        [modelName, await ((await fetch(baseUrl + '/openai-evals/' + modelName.replace('/', '--') + '/__index__.json')).json())])))
    const scores = computeRelativeOpenAiEvalsScores(reportsIndex)

    const tr = tableBodyE.insertRow()
    tr.classList.add('relative-average-score')
    tableBodyE.appendChild(tr)
    tr.insertCell().appendChild(createTextE('Relative average score'))
    for (const modelName of modelNames)
        createTableScoreCell(tr, createTextE(round(scores.averageRelativeScoresByModelName[modelName])))

    for (const [reportFilename, { spec }] of Object.entries(reportsIndex[modelNames[0]]).sort()) {
        const reportE = tableBodyE.insertRow()
        reportE.insertCell().appendChild(createLinkE(allowCharacterLineBreaks(spec.base_eval), { report: spec.eval_name, model: modelNames[0] }))

        const reportScores = scores.scoresByFilename[reportFilename]
        for (const modelName of modelNames) {
            let score = reportScores[modelName]
            if (typeof score == 'number')
                score = round(score)
            if (score == null)
                score = '-'
            createTableScoreCell(reportE, createLinkE(score, { report: spec.eval_name, model: modelName }))
        }
    }

    return reportsIndexE
}

export async function createV(baseUrl, parameters) {
    if (parameters.has('report') && parameters.has('model'))
        return createEvalReportsV(baseUrl, parameters.get('report'), parameters.get('model'), parameters.get('sample'))
    return await createEvalsIndexV(baseUrl)
}

export function computeRelativeOpenAiEvalsScores(openAiEvalsResults) {
    const modelNames = Object.keys(openAiEvalsResults)
    const firstModelName = modelNames[0]
    const firstModelResults = openAiEvalsResults[firstModelName]
    const reportsFilenames = Object.keys(firstModelResults)
    const reportsByFilename = Object.fromEntries(reportsFilenames.map(reportFilename => [reportFilename, Object.fromEntries(
        modelNames.map(modelName => [modelName, openAiEvalsResults[modelName][reportFilename]]))]))
    const scoresByFilename = Object.fromEntries(Object.entries(reportsByFilename).map(([reportFilename, reportByModelName]) => [reportFilename,
        Object.fromEntries(Object.entries(reportByModelName).map(([modelName, { spec, final_report }]) => [modelName, getScores(spec, final_report)]))]))
    const maxScoresByFilename = Object.fromEntries(Object.entries(scoresByFilename).map(([reportFilename, scoresByModelName]) =>
        [reportFilename, Math.max(...Object.values(scoresByModelName))]))
    const relativeScoresByFilename = Object.fromEntries(Object.entries(scoresByFilename).map(([reportFilename, scoresByModelName]) =>
        [reportFilename, Object.fromEntries(Object.entries(scoresByModelName).map(([modelName, score]) => [modelName, score / maxScoresByFilename[reportFilename]]))]))
    const includedReportsFilenames = reportsFilenames.filter(reportFilename => {
        if (firstModelResults[reportFilename].spec.run_config.eval_spec.cls == 'evals.elsuite.modelgraded.classify:ModelBasedClassify')
            return false
        for (const relativeModelScore of Object.values(relativeScoresByFilename[reportFilename]))
            if (isNaN(relativeModelScore))
                return false
        return true
    })
    const relativeScoresByFilenameFiltered = Object.fromEntries(Object.entries(relativeScoresByFilename)
        .filter(([reportFilename, _]) => includedReportsFilenames.includes(reportFilename)))
    const relativeScoresByModelName = modelNames.map(modelName =>
        [modelName, Object.values(relativeScoresByFilenameFiltered).map(relativeScores => relativeScores[modelName])])
    const averageRelativeScoresByModelName = Object.fromEntries(relativeScoresByModelName
        .map(([modelName, relativeScores]) => [modelName, relativeScores.reduce((a, b) => a + b, 0) / relativeScores.length]))
    return { averageRelativeScoresByModelName, scoresByFilename }
}
