'use strict'

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
    return [Object.entries(finalReport).map(e => e[0] + ': ' + e[1]).sort(), [...samples.entries()].map(([sampleId, sample]) => [sampleId, [
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
                for (let i = 1; i <= Object.keys(sample.sampling.prompt).length; i++)
                    add(sample.sampling.prompt['input' + i], sample.sampling.sampled['completion' + i], i)
            } else {
                add(sample.sampling.prompt.input, sample.sampling.sampled.completion)
            }

            return [
                ...ret,
                createExplanationTextE('The model was then asked to evaluate its own answer' + (usesMultipleInputs ? 's' : '') + ':'),
            ]
        })(),

        createConversationE(sample.sampling.info.prompt),

        createExplanationTextE('The model responded to this evaluation as follows:'),
        createConversationE([{ role: 'assistant', content: sample.sampling.info.sampled }]),

        createExplanationTextE('This was ' + (sample.sampling.info.invalid_choice ? 'an invalid' : 'a valid')
            + ' response to the evaluation. The resulting score is ' + sample.sampling.info.score + '.')
    ]])]
}

function showDataFromMatch(finalReport, samples) {
    return [[
        'Accuracy: ' + finalReport.accuracy,
    ], [...samples.entries()].map(([sampleId, sample]) => [sampleId, [
        createExplanationTextE('The model got the following start of a conversation as input:'),
        createConversationE(sample.match.prompt),

        createExplanationTextE('The model answered in the following way:'),
        createConversationE([{ role: 'assistant', content: sample.match.sampled }]),

        createExplanationTextE('The following answer'
            + (sample.match.options.length === 1 ? ' was' : 's were')
            + ' expected:'),
        ...sample.match.options.map(e => createConversationE([{ role: 'assistant', content: e }])),

        createExplanationTextE('The model answer was judged as ' + (sample.match.correct ? 'correct.' : 'incorrect.')),
    ]])]
}

function showDataFromFuzzyMatch(finalReport, samples) {
    return [[
        'Accuracy: ' + finalReport.accuracy,
        'F1 score: ' + finalReport.f1_score,
    ], [...samples.entries()].map(([sampleId, sample]) => [sampleId, [
        createExplanationTextE('The model got the following start of a conversation as input:'),
        createConversationE(sample.match.test_sample.input),

        createExplanationTextE('The following answer'
            + (sample.match.test_sample.ideal.length === 1 ? ' was' : 's were')
            + ' expected:'),
        ...sample.match.test_sample.ideal.map(e => createConversationE([{ role: 'assistant', content: e }])),

        createExplanationTextE('The model answered in the following way:'),
        createConversationE([{ role: 'assistant', content: sample.match.sampled }]),

        createExplanationTextE('The model answer was judged as ' + (sample.match.correct ? 'correct.' : 'incorrect.')),
    ]])]
}

function showDataFromIncludes(finalReport, samples) {
    return [[
        'Accuracy: ' + finalReport.accuracy
    ], [...samples.entries()].map(([sampleId, sample]) => [sampleId, [
        createExplanationTextE('The model got the following start of a conversation as input:'),
        createConversationE(sample.match.prompt),

        createExplanationTextE('The following answer'
            + (typeof sample.match.expected === 'string' ? ' was' : 's were')
            + ' expected:'),
        ...(typeof sample.match.expected === 'string' ? [sample.match.expected] : sample.match.expected)
            .map(e => createConversationE([{ role: 'assistant', content: e }])),

        createExplanationTextE('The model answered in the following way:'),
        createConversationE([{ role: 'assistant', content: sample.match.sampled }]),

        createExplanationTextE('The model answer was judged as ' + (sample.match.correct ? 'correct.' : 'incorrect.')),
    ]])]
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

    const [finalReportLines, mappedSamples] = showData(spec, finalReport, reportDataBySampleId)

    const finalReportInformationE = document.createElement('div')
    finalReportInformationE.classList.add('final-report-information')
    finalReportInformationE.appendChild(createExplanationTextE('Name: ' + spec.base_eval))
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

async function showReportIndex(url) {
    const reportIndex = await (await fetch(url)).json()

    const reportIndexE = document.createElement('ul')
    for (const [reportFilename, { spec, final_report: finalReport }] of Object.entries(reportIndex)) {
        const reportE = document.createElement('li')
        reportIndexE.appendChild(reportE)

        const reportLinkE = document.createElement('a')
        reportLinkE.textContent = reportFilename
        reportLinkE.href = '#https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/' + reportFilename
        reportE.appendChild(reportLinkE)
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
