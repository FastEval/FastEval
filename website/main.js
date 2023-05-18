'use strict'

function round(num) {
    return Math.round(num * 10000) / 10000
}

function allowCharacterLineBreaks(text, characters = ['/', '_']) {
    let out = text
    for (const char of characters)
        out = out.replaceAll(char, char + '\u200b')
    return out
}

function createExplanationTextE(text) {
    const explanationTextE = document.createElement('span')
    explanationTextE.textContent = text
    return explanationTextE
}

function createConversationItemE(role, text) {
    const containerE = document.createElement('div')
    containerE.classList.add('conversation-item')
    containerE.classList.add('conversation-item-' + role)

    const roleE = document.createElement('span')
    roleE.classList.add('conversation-item__role')
    roleE.textContent = role.charAt(0).toUpperCase() + role.slice(1)
    containerE.appendChild(roleE)

    const contentE = document.createElement('pre')
    contentE.textContent = text
    containerE.appendChild(contentE)

    return containerE
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
                ret.push(createExplanationTextE('The model got the following start of a conversation as input' + indexIndictator))
                ret.push(createConversationE(prompt))
                ret.push(createExplanationTextE('The model answered in the following way:' + indexIndictator))
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
                createExplanationTextE('The model was then asked to evaluate its own answer' + (usesMultipleInputs ? 's' : '') + ':'),
            ]
        })(),

        createConversationE(sample.sampling.info.prompt),

        createExplanationTextE('The model responded to this evaluation as follows:'),
        createConversationE([{ role: 'assistant', content: sample.sampling.info.sampled }]),

        createExplanationTextE('This was ' + (sample.sampling.info.invalid_choice ? 'an invalid' : 'a valid') + ' response to the evaluation.'
            + (sample.sampling.info.score !== null ? (' The resulting score is ' + sample.sampling.info.score + '.') : '')),
    ]])]
}

function createAnswersE(options) {
    if (typeof options === 'string')
        options = [options]
    const onlySingleOption = options.length === 1

    return [
        createExplanationTextE('The following answer' + (onlySingleOption ? ' was' : 's were') + ' expected:'),
        ...options.map(e => createConversationE([{ role: 'assistant', content: e }])),
    ]
}

function createMatchEs(prompt, correctAnswers, sampledModelAnswers, modelAnswerIsCorrect) {
    return [
        createExplanationTextE('The model got the following start of a conversation as input:'),
        createConversationE(prompt),

        ...createAnswersE(correctAnswers),

        createExplanationTextE('The model answered in the following way:'),
        createConversationE([{ role: 'assistant', content: sampledModelAnswers }]),

        createExplanationTextE('The model answer was judged to be ' + (modelAnswerIsCorrect ? 'correct.' : 'incorrect.')),
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
                createExplanationTextE('The SacreBLEU score is ' + round(sample.metrics.sacrebleu_sentence_score) + '.'),
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

function createSamplesV(mappedSamples, reportUrlWithoutHash) {
    const containerE = document.createElement('div')
    containerE.classList.add('samples')

    for (const [sampleId, mappedSample] of mappedSamples) {
        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        containerE.appendChild(sampleE)

        const sampleIdE = document.createElement('a')
        sampleIdE.textContent = 'ID: ' + sampleId
        const sampleUrl = new URL(location.toString())
        sampleUrl.hash = reportUrlWithoutHash
        sampleUrl.hash += '#' + sampleId
        sampleIdE.href = sampleUrl

        sampleE.append(
            sampleIdE,
            ...mappedSample,
        )
    }

    return containerE
}

function createSelectedModelReportV(reportUrl, report) {
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

    const reportUrlWithoutHash = new URL(reportUrl)
    reportUrlWithoutHash.hash = ''

    const hash = new URL(reportUrl).hash
    const onlyShowSingleSample = hash === '' ? null : hash.substring(1)

    if (!onlyShowSingleSample)
        selectedModelInformationE.append(...finalReportLines.map(line => createExplanationTextE(line)))

    const samplesE = onlyShowSingleSample
        ? createSamplesV(mappedSamples.filter(s => s[0] === onlyShowSingleSample), reportUrlWithoutHash)
        : createSamplesV(mappedSamples, reportUrlWithoutHash)
    containerE.appendChild(samplesE)

    return containerE
}

async function createEvalReportsV(reportUrls) {
    const containerE = document.createElement('div')

    const firstReport = await (await fetch(reportUrls[0])).text()
    const firstSpec = JSON.parse(firstReport.split('\n')[0]).spec

    const finalReportInformationE = document.createElement('div')
    containerE.appendChild(finalReportInformationE)
    finalReportInformationE.classList.add('final-report-information')
    finalReportInformationE.appendChild(createExplanationTextE('Name: ' + firstSpec.eval_name))
    finalReportInformationE.appendChild(createExplanationTextE('Evaluation method: ' + firstSpec.run_config.eval_spec.cls.split(':').slice(-1)))

    if (reportUrls.length === 1) {
        containerE.appendChild(createExplanationTextE('Model: ' + reportUrls[0].split('/').slice(-2, -1)))
        containerE.appendChild(createSelectedModelReportV(reportUrls[0], firstReport))
        return containerE
    }

    const modelSelectV = document.createElement('div')
    containerE.appendChild(modelSelectV)
    modelSelectV.appendChild(createExplanationTextE('Model: '))
    const modelSelectE = document.createElement('select')
    modelSelectV.appendChild(modelSelectE)
    for (const url of reportUrls) {
        const optionE = document.createElement('option')
        optionE.value = url
        optionE.textContent = url.split('/').slice(-2, -1)[0].replace('--', '/')
        modelSelectE.appendChild(optionE)
    }

    let reportV = createSelectedModelReportV(reportUrls[0], firstReport)
    containerE.appendChild(reportV)

    const fetchedReports = new Map()
    fetchedReports.set(reportUrls[0], firstReport)

    modelSelectE.addEventListener('change', async () => {
        const newReportUrl = modelSelectE.value
        const newReport = fetchedReports.get(newReportUrl) ?? await (await fetch(newReportUrl)).text()
        fetchedReports.set(newReportUrl, newReport)
        const newReportV = createSelectedModelReportV(newReportUrl, newReport)
        containerE.replaceChild(newReportV, reportV)
        reportV = newReportV
    })

    return containerE
}

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

async function createEvalsIndexV(urls) {
    const reportsIndexE = document.createElement('table')
    const tableHeadE = reportsIndexE.createTHead().insertRow()
    const tableBodyE = reportsIndexE.createTBody()
    tableHeadE.insertCell().appendChild(createExplanationTextE('Eval name'))

    for (const url of urls)
        tableHeadE.insertCell().appendChild(createExplanationTextE(allowCharacterLineBreaks(url.split('/').slice(-2, -1)[0].replace('--', '/'))))

    const tr = tableBodyE.insertRow()
    tr.classList.add('relative-average-score')
    tableBodyE.appendChild(tr)
    tr.insertCell().appendChild(createExplanationTextE('Relative average score'))
    const relativeAverageScoreEs = urls.map(url => tr.insertCell())

    const reportsIndex = Object.fromEntries(await Promise.all(urls.map(async url => [url, await ((await fetch(url)).json())])))

    const modelPerformances = Array(urls.length).fill(0)
    let numModelScores = 0

    for (const [reportFilename, { spec }] of Object.entries(reportsIndex[urls[0]]).sort()) {
        const reportE = tableBodyE.insertRow()

        const evalNameE = document.createElement('a')
        evalNameE.textContent = allowCharacterLineBreaks(spec.base_eval)
        evalNameE.href = '#' + urls.map(url => url.replace('__index__.json', reportFilename)).join(',')
        reportE.insertCell().appendChild(evalNameE)

        const scores = Object.fromEntries(urls.map(url => [url, getScores(spec, reportsIndex[url][reportFilename].final_report)]))
        const maxScore = Math.max(...Object.values(scores))
        let otherScoresAreNumbers = true
        for (const [i, url] of urls.entries()) {
            const scoreE = document.createElement('a')
            const score = scores[url]
            scoreE.textContent = score ?? '-'
            if (score === maxScore)
                scoreE.classList.add('max-score')
            scoreE.href = '#' + url.replace('__index__.json', reportFilename)
            reportE.insertCell().appendChild(scoreE)

            if (spec.run_config.eval_spec.cls === 'evals.elsuite.modelgraded.classify:ModelBasedClassify')
                continue

            if (typeof score === 'number') {
                if (!otherScoresAreNumbers)
                    throw new Error()
                modelPerformances[i] += (maxScore === 0) ? 1 : (score / maxScore)
                if (i === 0)
                    numModelScores++
            } else {
                otherScoresAreNumbers = false
            }
        }
    }

    for (let i = 0; i < urls.length; i++)
        relativeAverageScoreEs[i].appendChild(createExplanationTextE(round(modelPerformances[i] / numModelScores)))

    return reportsIndexE
}

async function createReportsV(urls) {
    urls = urls.split('\n').map(url => url.trim())

    const isIndexUrl = url => url.endsWith('__index__.json')
    const allUrlsAreIndexUrls = urls.every(isIndexUrl)
    const someUrlIsIndexUrl = urls.some(isIndexUrl)

    if (someUrlIsIndexUrl && !allUrlsAreIndexUrls)
        alert('Either all URLs must be index urls or none of them.')

    if (someUrlIsIndexUrl)
        return await createEvalsIndexV(urls)
    else
        return await createEvalReportsV(urls)
}

async function createMainV() {
    const containerE = document.createElement('div')

    const showUrlsE = document.createElement('button')
    showUrlsE.textContent = 'Click here to show & edit report urls'
    containerE.appendChild(showUrlsE)

    const urlsE = document.createElement('textarea')
    urlsE.spellcheck = false
    urlsE.rows = 7
    urlsE.value = location.hash.substring(1).replaceAll(',', '\n') || (
          'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/OpenAssistant--pythia-12b-sft-v8-7k-steps/__index__.json\n'
        + 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/oasst-rlhf-2-llama-30b-7k-steps-wrongly-used/__index__.json\n'
        + 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/OpenAssistant--oasst-rlhf-3-llama-30b-5k-steps/__index__.json\n'
        + 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/oasst-sft-7-llama-30b/__index__.json\n'
        + 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/oasst-sft-7e3-llama-30b/__index__.json\n'
        + 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/OpenAssistant--llama-30b-sft-v8-2.5k-steps/__index__.json\n'
        + 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/reports/gpt-3.5-turbo/__index__.json'
    )

    showUrlsE.addEventListener('click', () => {
        containerE.removeChild(showUrlsE)
        containerE.prepend(urlsE)
    })

    containerE.appendChild(await createReportsV(urlsE.value))

    window.addEventListener('hashchange', () => {
        location.reload()
    })

    urlsE.addEventListener('change', () => {
        location.hash = urlsE.value.replaceAll('\n', ',')
    })

    return containerE
}

createMainV().then(mainV => document.body.appendChild(mainV))
