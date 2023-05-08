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
    const finalReportLines = [
        'Score: ' + finalReport.score,
        '#correct: ' + (finalReport['counts/Y'] ?? 0),
        '#incorrect: ' + (finalReport['counts/N'] ?? 0),
        '#invalid: ' + (finalReport['counts/__invalid__'] ?? 0),
    ]

    const samplesE = document.createElement('div')
    for (const [sampleId, sample] of samples.entries()) {
        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        samplesE.appendChild(sampleE)

        sampleE.append(
            createUnderlinedExplanationTextE('ID: ' + sampleId),

            createExplanationTextE('The model got the following start of a conversation as input:'),
            createConversationE(sample.sampling.prompt.input),

            createExplanationTextE('The model answered in the following way:'),
            createConversationE([{ role: 'assistant', content: sample.sampling.sampled.completion }]),

            createExplanationTextE('The model was then asked to evaluate its own answer:'),
            createConversationE(sample.sampling.info.prompt),

            createExplanationTextE('The model responded to this evaluation as follows:'),
            createConversationE([{ role: 'assistant', content: sample.sampling.info.sampled }]),

            createExplanationTextE('This was ' + (sample.sampling.info.invalid_choice ? 'an invalid' : 'a valid')
                + ' response to the evaluation. The resulting score is ' + sample.sampling.info.score + '.')
        )
    }

    return [finalReportLines, samplesE]
}

function showDataFromMatch(finalReport, samples) {
    const finalReportLines = ['Accuracy: ' + finalReport.accuracy]

    const samplesE = document.createElement('div')
    for (const [sampleId, sample] of samples) {
        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        samplesE.appendChild(sampleE)

        sampleE.append(
            createUnderlinedExplanationTextE('ID: ' + sampleId),

            createExplanationTextE('The model answered in the following way:'),
            createConversationE([{ role: 'assistant', content: sample.match.sampled }]),

            createExplanationTextE('The following answer'
                + (sample.match.options.length === 1 ? ' was' : 's were')
                + ' expected:'),
            ...sample.match.options.map(e => createConversationE([{ role: 'assistant', content: e }])),

            createExplanationTextE('The model answer was judged as ' + (sample.match.correct ? 'correct.' : 'incorrect.')),
        )
    }

    return [finalReportLines, samplesE]
}

function showDataFromFuzzyMatch(finalReport, samples) {
    const finalReportLines = [
        'Accuracy: ' + finalReport.accuracy,
        'F1 score: ' + finalReport.f1_score,
    ]

    console.log(samples)

    const samplesE = document.createElement('div')
    for (const [sampleId, sample] of samples) {
        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        samplesE.appendChild(sampleE)

        sampleE.append(
            createUnderlinedExplanationTextE('ID: ' + sampleId),

            createExplanationTextE('The model got the following start of a conversation as input:'),
            createConversationE(sample.match.test_sample.input),

            createExplanationTextE('The following answer'
                + (sample.match.test_sample.ideal.length === 1 ? ' was' : 's were')
                + ' expected:'),
            ...sample.match.test_sample.ideal.map(e => createConversationE([{ role: 'assistant', content: e }])),

            createExplanationTextE('The model answered in the following way:'),
            createConversationE([{ role: 'assistant', content: sample.match.sampled }]),

            createExplanationTextE('The model answer was judged as ' + (sample.match.correct ? 'correct.' : 'incorrect.')),
        )
    }

    return [finalReportLines, samplesE]
}

function createDataE(spec, finalReport, samples) {
    switch (spec.run_config.eval_spec.cls) {
        case 'evals.elsuite.modelgraded.classify:ModelBasedClassify':
            return showDataFromModelBasedClassify(finalReport, samples)
        case 'evals.elsuite.basic.match:Match':
            return showDataFromMatch(finalReport, samples)
        case 'evals.elsuite.basic.fuzzy_match:FuzzyMatch':
            return showDataFromFuzzyMatch(finalReport, samples)
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

    const [finalReportLines, samplesE] = createDataE(spec, finalReport, reportDataBySampleId)

    const finalReportInformationE = document.createElement('div')
    finalReportInformationE.classList.add('final-report-information')
    finalReportInformationE.appendChild(createExplanationTextE('Name: ' + spec.base_eval))
    finalReportInformationE.append(...finalReportLines.map(line => createExplanationTextE(line)))

    samplesE.classList.add('samples')

    return [finalReportInformationE, samplesE]
}

function main() {
    const containerE = document.createElement('div')
    document.body.appendChild(containerE)

    const urlE = document.createElement('input')
    // urlE.value = 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/coqa-closedqa-conciseness.dev.v0.json'
    // urlE.value = 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/coqa-match.dev.v0.json'
    urlE.value = 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/test-fuzzy-match.s1.simple-v0.json'
    containerE.appendChild(urlE)

    const reportE = document.createElement('div')
    reportE.classList.add('report')
    containerE.appendChild(reportE)

    urlE.addEventListener('change', () => {
        showDataFromReportUrl(urlE.value).then(reportEs => reportE.replaceChildren(...reportEs))
    })

    showDataFromReportUrl(urlE.value).then(reportEs => reportE.replaceChildren(...reportEs))
}

main()
