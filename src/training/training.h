#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
#include "data/corpus_sqlite.h"
#endif
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

template <class ModelWrapper>
class Train : public ModelTask {
private:
  Ptr<Options> options_;

public:
  Train(Ptr<Options> options) : options_(options) {}

  void run() override {
    using namespace data;

    Ptr<CorpusBase> dataset;
    if(!options_->get<std::string>("sqlite").empty())
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
      dataset = New<CorpusSQLite>(options_);
#else
      ABORT("SqLite presently not supported on Windows");
#endif
    else
      dataset = New<Corpus>(options_);

    dataset->prepare();

    Ptr<BatchStats> stats;
    if(options_->get<bool>("mini-batch-fit")) {
      LOG(info,
          "[batching] Collecting statistics for batch fitting with step size "
          "{}",
          options_->get<size_t>("mini-batch-fit-step"));
      // @TODO, better fake batch with vocabulary
      auto model = New<ModelWrapper>(options_);
      THREAD_GUARD(stats = model->collectStats());
      LOG(info, "[batching] Done");
    }

    auto trainState = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, trainState);

    if((options_->has("valid-sets") || options_->has("valid-script-path"))
       && SchedulingParameter::parse(options_->get<std::string>("valid-freq"))) {
      for(auto validator : Validators(dataset->getVocabs(), options_))
        scheduler->addValidator(validator);
    }

    Ptr<CorpusBatchGenerator> batchGenerator;
    auto model = New<ModelWrapper>(options_);
    // gap-training
    // temporailly for mod-training
    auto graphs = model->collectGraphs();
    batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats, graphs);
    //if (options_->has("gap-training")){
    //  auto graphs = model->collectGraphs();
    //  batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats, graphs);
    //}else
    //  batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats);

    // put current batch number into option
    //options_->set("current-batch", trainState->batches);
    scheduler->registerTrainingObserver(batchGenerator);
    //
    model->setScheduler(scheduler);
    model->load();

    // @TODO: shuffle_ as a private attribute in BG
    auto shuffle = !options_->get<bool>("no-shuffle");
    bool restored = !options_->get<bool>("no-restore-corpus")
                    && batchGenerator->restore(trainState, shuffle);

    scheduler->started();
    while(scheduler->keepGoing()) {
      if(!restored)
        batchGenerator->prepare(shuffle);
      restored = false;

      // @TODO: try to use for(auto ...)
      for(auto batchIt = std::begin(*batchGenerator);
          batchIt != std::end(*batchGenerator) && scheduler->keepGoing();
          batchIt++) {
        model->update(*batchIt);
        //if (options_->has("sr-freq-file")){
        //  std::string schedule = options_->get<std::vector<std::string>>("sr-freq-file")[4];
        //  if (schedule == "mod")
        //    model->calculate_mod();
        //}
      }

      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();

    model->finalize();

    // Avoid saving the model twice if it has been loaded and training did not
    // progress
    if(!trainState->loaded)
      model->save(true);
  }
};
}  // namespace marian
