#pragma once

#include "data/types.h"
#include "hypothesis.h"

#include <queue>

namespace marian {

// search grid of one batch entry
class History {
private:
  // one hypothesis of a full sentence (reference into search grid)
  struct SentenceHypothesisCoord {
    bool operator<(const SentenceHypothesisCoord& hc) const { return normalizedPathScore < hc.normalizedPathScore; }

    size_t i; // last time step of this sentence hypothesis
    size_t j; // which beam entry
    float normalizedPathScore; // length-normalized sentence score
    size_t ret_code;

    //SentenceHypothesisCoord(size_t I, size_t J, float Score, size_t RET){
    //  i = I;
    //  j = J;
    //  normalizedPathScore = Score;
    //  ret_code = RET;
    //}

  };

public:
  History(size_t lineNo, float alpha = 1.f, float wp_ = 0.f);

  float LengthPenalty(size_t length) { return std::pow((float)length, alpha_); }
  float WordPenalty(size_t length) { return wp_ * (float)length; }

  void Add(const Beam& beam, Word trgEosId, bool last = false, bool cutoff = false) {
    if(beam.back()->GetPrevHyp() != nullptr) {
      for(size_t j = 0; j < beam.size(); ++j)
        if(beam[j]->GetWord() == trgEosId || last) {
          float pathScore = (beam[j]->GetPathScore() - WordPenalty(history_.size()))
                       / LengthPenalty(history_.size());
          topHyps_.push({history_.size(), j, pathScore, cutoff});
          // std::cerr << "Add " << history_.size() << " " << j << " " << pathScore
          // << std::endl;
        }
    }
    history_.push_back(beam);
  }

  void Remove(const size_t num){
    history_ = std::vector<Beam>(history_.begin(), history_.end() - num);
    // every time remove rows of beams, clear topHyps.
    std::vector<SentenceHypothesisCoord> t_container;
    while(!topHyps_.empty()){
      t_container.emplace_back(topHyps_.top());
      topHyps_.pop();
    }
    for (auto &t: t_container)
      if (t.i <= history_.size())
        topHyps_.push(t);
  }

  Beam LastBeam(){
    return history_.back();
  }

  size_t size() const { return history_.size(); } // number of time steps

  NBestList NBest(size_t n) const {
    NBestList nbest;
    for (auto topHypsCopy = topHyps_; nbest.size() < n && !topHypsCopy.empty(); topHypsCopy.pop()) {
      auto bestHypCoord = topHypsCopy.top();

      const size_t start = bestHypCoord.i; // last time step of this hypothesis
      const size_t j     = bestHypCoord.j; // which beam entry
      Ptr<Hypothesis> bestHyp = history_[start][j];
      // float c = bestHypCoord.normalizedPathScore;
      // std::cerr << "h: " << start << " " << j << " " << c << std::endl;

      // trace back best path
      Words targetWords = bestHyp->TracebackWords();

      // note: bestHyp->GetPathScore() is not normalized, while bestHypCoord.normalizedPathScore is
      nbest.emplace_back(targetWords, bestHyp, bestHypCoord.normalizedPathScore, bestHypCoord.ret_code);
    }
    return nbest;
  }

  Result Top() const { return NBest(1)[0]; }

  size_t GetLineNum() const { return lineNo_; }

private:
  std::vector<Beam> history_; // [time step][index into beam] search grid
  std::priority_queue<SentenceHypothesisCoord> topHyps_; // all sentence hypotheses (those that reached eos), sorted by score
  size_t lineNo_;
  float alpha_;
  float wp_;
};

typedef std::vector<Ptr<History>> Histories;
}  // namespace marian
