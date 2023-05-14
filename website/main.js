'use strict'

function round(num) {
    return Math.round(num * 10000) / 10000
}

function createExplanationTextE(text) {
    const explanationTextE = document.createElement('span')
    explanationTextE.textContent = text
    return explanationTextE
}

function createUnderlinedExplanationTextE(text) {
    const explanationTextE = createExplanationTextE(text)
    explanationTextE.classList.add('underlined-explanation-text')
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
        default:
            throw new Error()
    }
}

async function showDataFromReportUrl(reportUrl) {
    const report = await (await fetch(reportUrl)).text()
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

    const finalReportInformationE = document.createElement('div')
    finalReportInformationE.classList.add('final-report-information')
    finalReportInformationE.appendChild(createExplanationTextE('Name: ' + spec.eval_name))
    finalReportInformationE.appendChild(createExplanationTextE('Evaluation method: ' + spec.run_config.eval_spec.cls.split(':').slice(-1)))
    finalReportInformationE.append(...finalReportLines.map(line => createExplanationTextE(line)))

    const samplesE = document.createElement('div')
    samplesE.classList.add('samples')
    for (const [sampleId, mappedSample] of mappedSamples) {
        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        samplesE.appendChild(sampleE)

        sampleE.append(
            createUnderlinedExplanationTextE('ID: ' + sampleId),
            ...mappedSample,
        )
    }

    return [finalReportInformationE, samplesE]
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
        default:
            throw new Error()
    }
}

async function showReportIndex(url) {
    const reportIndex = await (await fetch(url)).json()

    const reportIndexE = document.createElement('table')
    const tableHeadE = reportIndexE.createTHead().insertRow()
    tableHeadE.insertCell().appendChild(createExplanationTextE('Eval name'))
    tableHeadE.insertCell().appendChild(createExplanationTextE('Score'))

    for (const [reportFilename, { spec, final_report: finalReport }] of Object.entries(reportIndex).sort()) {
        const reportE = reportIndexE.insertRow()

        const evalNameE = createExplanationTextE(spec.eval_name)
        reportE.insertCell().appendChild(evalNameE)

        const scoresE = document.createElement('a')
        scoresE.textContent = getScores(spec, finalReport) ?? '-'
        scoresE.href = '#https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/' + reportFilename
        reportE.insertCell().appendChild(scoresE)
    }

    return [reportIndexE]
}

function showReportIndexOrUrl(reportE, url) {
    if (url.endsWith('__index__.json'))
        showReportIndex(url).then(reportIndexEs => reportE.replaceChildren(...reportIndexEs))
    else
        showDataFromReportUrl(url).then(reportEs => reportE.replaceChildren(...reportEs))
}

function main() {
    const containerE = document.createElement('div')
    document.body.appendChild(containerE)

    const urlE = document.createElement('input')
    urlE.value = location.hash.substring(1) || 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/__index__.json'
    containerE.appendChild(urlE)

    const reportE = document.createElement('div')
    reportE.classList.add('report')
    containerE.appendChild(reportE)

    urlE.addEventListener('change', () => {
        showReportIndexOrUrl(reportE, urlE.value)
    })

    showReportIndexOrUrl(reportE, urlE.value)

    window.addEventListener('hashchange', () => {
        location.reload()
    })
}

main()
