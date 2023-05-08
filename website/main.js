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

function showDataFromModelBasedClassify(finalReport, data) {
    const reportDataBySampleId = new Map()
    for (const event of data) {
        const item = reportDataBySampleId.get(event.sample_id) ?? {}
        item[event.type] = event.data
        reportDataBySampleId.set(event.sample_id, item)
    }

    const containerE = document.createElement('div')
    containerE.classList.add('final-report-with-data')

    console.log(finalReport)

    const finalReportE = document.createElement('div')
    finalReportE.classList.add('explanations')
    finalReportE.appendChild(createExplanationTextE('#correct: ' + (finalReport['counts/Y'] ?? 0)))
    finalReportE.appendChild(createExplanationTextE('#incorrect: ' + (finalReport['counts/N'] ?? 0)))
    finalReportE.appendChild(createExplanationTextE('#invalid: ' + (finalReport['counts/__invalid__'] ?? 0)))
    containerE.appendChild(finalReportE)

    const samplesE = document.createElement('div')
    samplesE.classList.add('samples')
    containerE.appendChild(samplesE)

    for (const [sampleId, sample] of reportDataBySampleId.entries()) {
        const initialPrompt = sample.sampling.prompt.input
        const initialPromptGeneratedOutput = sample.sampling.sampled.completion
        const evaluationPrompt = sample.sampling.info.prompt
        const evaluationGeneratedOutput = sample.sampling.info.sampled
        const evaluationScore = sample.sampling.info.score
        const evaluationIsInvalid = sample.sampling.info.invalid_choice

        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        samplesE.appendChild(sampleE)

        sampleE.appendChild(createUnderlinedExplanationTextE('ID: ' + sampleId))

        sampleE.appendChild(createExplanationTextE('The model got the following start of a conversation as input:'))
        sampleE.appendChild(createConversationE(initialPrompt))

        sampleE.appendChild(createExplanationTextE('The model answered in the following way:'))
        sampleE.appendChild(createConversationE([{ role: 'assistant', content: initialPromptGeneratedOutput }]))

        sampleE.appendChild(createExplanationTextE('The model was then asked to evaluate its own answer:'))
        sampleE.appendChild(createConversationE(evaluationPrompt))

        sampleE.appendChild(createExplanationTextE('The model responded to this evaluation as follows:'))
        sampleE.appendChild(createConversationE([{ role: 'assistant', content: evaluationGeneratedOutput }]))

        sampleE.appendChild(createExplanationTextE('This was ' + (evaluationIsInvalid ? 'an invalid' : 'a valid')
            + ' response to the evaluation. The resulting score is ' + evaluationScore + '.'))
    }

    return containerE
}

function createDataE(spec, finalReport, data) {
    switch (spec.run_config.eval_spec.cls) {
        case 'evals.elsuite.modelgraded.classify:ModelBasedClassify':
            return showDataFromModelBasedClassify(finalReport, data)
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

    return [
        createExplanationTextE('Name: ' + spec.base_eval),
        createExplanationTextE('Score: ' + finalReport.score),
        createDataE(spec, finalReport, data),
    ]
}

function main() {
    const containerE = document.createElement('div')
    document.body.appendChild(containerE)

    const urlE = document.createElement('input')
    urlE.value = 'https://raw.githubusercontent.com/tju01/oasst-openai-evals/main/runs/coqa-closedqa-conciseness.dev.v0.json'
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
